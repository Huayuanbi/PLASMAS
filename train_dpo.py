"""SFT-initialized DPO with optional chosen NLL and prompt-only graph ranking."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from plasmas.qwen_topology_dpo import TopologyDPOExample
from plasmas.qwen_graph_dpo import GraphHead, graph_targets, graph_scores, graph_pair_loss, prompt_batch, chosen_token_nll
from training_utils import DPOCollator, FlatPairDataset, move_batch, policy_and_reference_logps, dpo_losses


def pair_weight(row, cost_multiplier=1.):
    weight=float(row['weight']) * (cost_multiplier if row['category']=='cost' else 1.)
    if not math.isfinite(weight) or weight<=0:
        raise ValueError('invalid pair weight')
    return weight


def prune_optimizer_states(directory, keep, prefix='epoch'):
    """Only discard old runtime.pt from completed, real epoch directories.

    Every adapter, graph head, tokenizer, and config remains available for inference.
    A positive keep value is opt-in for newly launched storage-limited sweeps.
    """
    if keep<=0:
        return []
    checkpoints=[]
    if prefix not in ('epoch','step'):
        raise ValueError('invalid checkpoint prefix')
    for path in directory.glob(prefix+'*/runtime.pt'):
        match=re.fullmatch(prefix+r'([0-9]+)',path.parent.name)
        if (match and not path.is_symlink() and not path.parent.is_symlink()
                and (path.parent/'COMPLETE').is_file()):
            checkpoints.append((int(match[1]),path))
    removed=[]
    for _,path in sorted(checkpoints)[:-keep]:
        path.unlink();removed.append(str(path))
        if prefix=='step':
            # Rolling step checkpoints are for recovery, not the epoch sweep.
            # Preserve all epoch adapters; retire only our old step weights.
            for name in ['adapter_model.safetensors','graph_head.pt','COMPLETE']:
                old=path.parent/name
                if old.is_file() and not old.is_symlink():
                    old.unlink();removed.append(str(old))
    return removed


def latest_checkpoint(directory):
    candidates=[]
    for marker in directory.glob('*/COMPLETE'):
        if not re.fullmatch(r'(epoch|step)[0-9]+',marker.parent.name):continue
        cursor=marker.parent/'cursor.json'
        if (marker.parent/'runtime.pt').exists() and cursor.exists():
            state=json.loads(cursor.read_text())
            candidates.append((state['step'],not state.get('next_micro',0),marker.parent))
    if candidates:return max(candidates)[2]
    # Backward compatibility for epoch-only checkpoints written before cursor.json.
    old=[p.parent for p in directory.glob('epoch*/COMPLETE') if (p.parent/'runtime.pt').exists()]
    return max(old,key=lambda p:int(p.name[5:])) if old else None


def save_checkpoint(path, model, tokenizer, head, state):
    """A failed write never advertises a complete checkpoint or replaces an old one."""
    if path.exists():raise ValueError(f'checkpoint already exists: {path}')
    staging=Path(tempfile.mkdtemp(prefix='.'+path.name+'.incomplete-',dir=path.parent))
    model.save_pretrained(staging);tokenizer.save_pretrained(staging)
    if head is not None:torch.save(head.state_dict(),staging/'graph_head.pt')
    torch.save(state,staging/'runtime.pt')
    (staging/'config.json').write_text(json.dumps(state['config'],indent=2))
    (staging/'cursor.json').write_text(json.dumps({k:state[k] for k in ['epoch','step','next_micro']}))
    (staging/'COMPLETE').touch()
    staging.rename(path)


def load_pairs(path, tokenizer, node_ids, cost_multiplier=1.):
    examples, targets, identities = [], [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        if row.get('prompt', prompt) != prompt:
            raise ValueError('prompt/tokenizer mismatch')
        if row['chosen'] == row['rejected'] or not 0 < float(row['weight']) < float('inf'):
            raise ValueError('invalid pair or weight')
        examples.append(TopologyDPOExample(prompt, row['chosen'], row['rejected'],
            pair_weight(row,cost_multiplier), row['category'], float(row['ranking_gap']), int(row['group_index'])))
        targets.append((graph_targets(row['chosen'], node_ids), graph_targets(row['rejected'], node_ids)))
        identities.add(prompt)
    if not examples:
        raise ValueError(f'empty pair file: {path}')
    return examples, targets, identities


def batch_terms(model, head, batch, targets, args):
    policy, reference = policy_and_reference_logps(model, batch, args.activation_offload_threshold)
    n = batch['weights'].numel()
    dpo, margin = dpo_losses(policy[:n], policy[n:], reference[:n], reference[n:], args.beta)
    sft = chosen_token_nll(policy[:n], batch['labels'][:n])
    structural, structure_margin = torch.zeros_like(dpo), torch.zeros_like(dpo)
    if head is not None:
        ids, mask, lengths = prompt_batch(batch)
        ctx = (torch.autograd.graph.save_on_cpu(pin_memory=False, device_type='cuda')
               if ids.is_cuda and args.activation_offload_threshold > 0 and ids.size(1) >= args.activation_offload_threshold
               else nullcontext())
        with ctx:
            # Qwen decoder only: no vocabulary logits, no completion tokens.
            hidden = model.get_base_model().model(input_ids=ids, attention_mask=mask,
                    use_cache=False, return_dict=True).last_hidden_state
            last = hidden[torch.arange(n, device=ids.device), lengths - 1]
            node_logits, edge_logits = head(last)
            selected = [targets[i] for i in batch['indices'].tolist()]
            def target(which):
                return tuple(torch.stack([x[which][j] for x in selected]).to(ids.device) for j in range(2))
            pos = graph_scores(node_logits, edge_logits, target(0), args.node_weight, args.edge_weight)
            neg = graph_scores(node_logits, edge_logits, target(1), args.node_weight, args.edge_weight)
            structural = graph_pair_loss(pos, neg, args.graph_margin, args.graph_temperature)
            structure_margin = pos - neg
    total = dpo + args.sft_weight * sft + args.graph_weight * structural
    return {'total': total, 'dpo': dpo, 'chosen_nll': sft, 'graph_pair': structural,
            'dpo_margin': margin,
            'dpo_accuracy': (margin > 0).float(), 'graph_accuracy': (structure_margin > 0).float(),
            'graph_margin': structure_margin}


@torch.no_grad()
def evaluate(model, head, loader, targets, args, categories=None):
    model.eval()
    if head is not None: head.eval()
    sums, weight = {}, 0.
    category_sums, category_weights = {}, {}
    for batch in loader:
        batch = move_batch(batch, torch.device(args.device))
        terms = batch_terms(model, head, batch, targets, args)
        w = batch['weights']
        weight += float(w.sum())
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.) + float((v * w).sum())
            if not torch.isfinite(v).all():
                raise FloatingPointError(f'nonfinite validation metric: {k}')
        if categories is not None:
            for i,index in enumerate(batch['indices'].tolist()):
                category = categories[index]
                category_weights[category] = category_weights.get(category,0.) + float(w[i])
                acc = category_sums.setdefault(category,{})
                for k,v in terms.items():
                    acc[k] = acc.get(k,0.) + float(v[i]*w[i])
    model.train()
    if head is not None: head.train()
    return {**{k: v / weight for k, v in sums.items()},
            **{f'{category}/{k}':v/category_weights[category]
               for category,acc in category_sums.items() for k,v in acc.items()}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-pairs', type=Path, required=True)
    p.add_argument('--validation-pairs', type=Path, required=True)
    p.add_argument('--sft-adapter', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume-from', type=Path)
    p.add_argument('--resume-latest',action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation-steps', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--sft-weight', type=float, default=.1)
    p.add_argument('--graph-weight', type=float, default=0.)
    p.add_argument('--cost-weight-multiplier',type=float,default=1.,
                   help='Training cost-pair weight multiplier; validation weights stay fixed across experiments.')
    p.add_argument('--keep-optimizer-checkpoints',type=int,default=0,
                   help='Keep only latest N runtime.pt files; 0 keeps all. All LoRA epochs remain.')
    p.add_argument('--save-every-steps',type=int,default=0,
                   help='Save a resumable checkpoint on optimizer boundaries (0 disables). Latest two step optimizer states retained.')
    p.add_argument('--graph-margin', type=float, default=0.)
    p.add_argument('--graph-temperature', type=float, default=1.)
    p.add_argument('--node-weight', type=float, default=1.)
    p.add_argument('--edge-weight', type=float, default=1.)
    p.add_argument('--node-ids', nargs='+', default=[f'agent_{i}' for i in range(6)])
    p.add_argument('--max-length', type=int, default=2560)
    p.add_argument('--activation-offload-threshold', type=int, default=1800)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--wandb-project', default='')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.resume_latest:
        if args.resume_from:raise ValueError('choose explicit resume or resume-latest, not both')
        args.resume_from=latest_checkpoint(args.output_dir)
    if min(args.beta, args.lr, args.graph_temperature, args.epochs, args.batch_size, args.gradient_accumulation_steps) <= 0:
        raise ValueError('positive hyperparameters required')
    if min(args.sft_weight, args.graph_weight, args.graph_margin, args.node_weight, args.edge_weight) < 0:
        raise ValueError('loss weights and margin must be nonnegative')
    if not math.isfinite(args.cost_weight_multiplier) or args.cost_weight_multiplier<=0 or args.keep_optimizer_checkpoints<0 or args.save_every_steps<0:
        raise ValueError('positive cost multiplier and nonnegative retention required')
    random.seed(args.seed); torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    train, train_targets, train_ids = load_pairs(args.train_pairs, tokenizer, args.node_ids,args.cost_weight_multiplier)
    val, val_targets, val_ids = load_pairs(args.validation_pairs, tokenizer, args.node_ids)
    if train_ids & val_ids:
        raise ValueError('train/validation prompt overlap')
    collator = DPOCollator(tokenizer, args.max_length)
    # Catch long completions and bad graphs before loading weights.
    for example in train + val: collator([(0, example)])
    print(json.dumps({'train_pairs': len(train), 'validation_pairs': len(val),
                      'sft_weight': args.sft_weight, 'graph_weight': args.graph_weight,
                      'cost_weight_multiplier':args.cost_weight_multiplier,
                      'training_category_weights':{c:sum(x.weight for x in train if x.category==c) for c in ['quality','cost']}}), flush=True)
    if args.dry_run: return
    identity = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [args.train_pairs, args.validation_pairs, args.sft_adapter/'adapter_model.safetensors', args.sft_adapter/'adapter_config.json']}
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    runtime = None
    if args.resume_from:
        if not (args.resume_from/'COMPLETE').is_file():
            raise ValueError('resume requires a completed epoch checkpoint')
        if not (args.resume_from/'runtime.pt').is_file():
            raise ValueError('optimizer state absent/pruned; resume from one of the latest full checkpoints')
        runtime = torch.load(args.resume_from/'runtime.pt', map_location='cpu', weights_only=False)
        compatible = set(config) - {'resume_from', 'output_dir', 'epochs', 'wandb_project', 'device','keep_optimizer_checkpoints','save_every_steps','resume_latest'}
        defaults={'cost_weight_multiplier':1.}
        if runtime['identity'] != identity or any(runtime['config'].get(k,defaults.get(k)) != config[k] for k in compatible):
            raise ValueError('resume data/model/loss configuration mismatch')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output directory is nonempty; use --resume-from or a new directory')
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device)
    # Merge SFT once, then train a new delta adapter. Disabling that adapter
    # yields the frozen SFT reference, not the original Qwen base.
    base = PeftModel.from_pretrained(base, args.sft_adapter).merge_and_unload()
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    if args.resume_from:
        model = PeftModel.from_pretrained(base, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=16, lora_alpha=32,
            lora_dropout=0., target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    head = GraphHead(base.config.hidden_size, len(args.node_ids)).to(args.device) if args.graph_weight else None
    if runtime and head is not None: head.load_state_dict(torch.load(args.resume_from/'graph_head.pt', weights_only=True, map_location=args.device))
    parameters = [p for p in model.parameters() if p.requires_grad] + ([] if head is None else list(head.parameters()))
    unexpected = [name for name, parameter in model.named_parameters()
                  if parameter.requires_grad and 'lora_' not in name]
    if unexpected:
        raise ValueError(f'base/reference parameters unexpectedly trainable: {unexpected[:5]}')
    print(json.dumps({'reference': 'frozen merged SFT',
                      'trainable_lora_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'trainable_graph_parameters': 0 if head is None else sum(p.numel() for p in head.parameters()),
                      'epochs': args.epochs, 'lr': args.lr, 'beta': args.beta}), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    if runtime: optimizer.load_state_dict(runtime['optimizer'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'config.json').write_text(json.dumps(config, indent=2))
    val_loader = DataLoader(FlatPairDataset(val), batch_size=args.batch_size, collate_fn=collator)
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=config)
    def log(row):
        print(json.dumps(row), flush=True)
        with (args.output_dir/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if wandb_run: wandb_run.log(row)
    start = runtime['epoch']+(0 if runtime.get('next_micro',0) else 1) if runtime else 1
    step = runtime['step'] if runtime else 0
    if runtime:
        torch.set_rng_state(runtime['rng'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(runtime['cuda_rng'])
    mean_weight = sum(x.weight for x in train)/len(train)
    val_categories = [row['category'] for row in map(json.loads,args.validation_pairs.read_text().splitlines())]
    if runtime is None:
        log({'epoch':0, **{'validation/'+k:v for k,v in evaluate(model,head,val_loader,val_targets,args,val_categories).items()}})
    for epoch in range(start, args.epochs+1):
        loader = DataLoader(FlatPairDataset(train), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed+epoch), collate_fn=collator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        continuing=runtime is not None and epoch==start and runtime.get('next_micro',0)>0
        skip_micro=runtime['next_micro'] if continuing else 0
        epoch_sums = dict(runtime.get('epoch_sums',{})) if continuing else {}
        epoch_weight = runtime.get('epoch_weight',0.) if continuing else 0.
        epoch_started = time.monotonic()
        def snapshot(next_micro):
            return {'epoch':epoch,'step':step,'next_micro':next_micro,
                'epoch_sums':epoch_sums,'epoch_weight':epoch_weight,
                'optimizer':optimizer.state_dict(),'identity':identity,'config':config,
                'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        for micro, batch in enumerate(loader):
            if micro<skip_micro:continue
            batch = move_batch(batch, torch.device(args.device))
            terms = batch_terms(model, head, batch, train_targets, args)
            loss = (terms['total'] * batch['weights'] / mean_weight).mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite joint loss')
            epoch_weight += float(batch['weights'].sum())
            for k,v in terms.items():
                epoch_sums[k] = epoch_sums.get(k,0.) + float((v.detach()*batch['weights']).sum())
            block_size = min(args.gradient_accumulation_steps, len(loader) - (micro//args.gradient_accumulation_steps)*args.gradient_accumulation_steps)
            (loss/block_size).backward()
            if (micro+1)%args.gradient_accumulation_steps==0 or micro+1==len(loader):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step+=1
                if args.save_every_steps and (step==1 or step%args.save_every_steps==0) and micro+1<len(loader):
                    save_checkpoint(args.output_dir/f'step{step}',model,tokenizer,head,snapshot(micro+1))
                    removed=prune_optimizer_states(args.output_dir,2,prefix='step')
                    if removed:print(json.dumps({'pruned_step_optimizer_states':removed}),flush=True)
                log({'epoch':epoch,'step':step,'grad_norm':float(norm),
                     **{'train_last_microbatch/'+k:float(v.detach().mean()) for k,v in terms.items()}})
        metrics = evaluate(model,head,val_loader,val_targets,args,val_categories)
        log({'epoch':epoch,'step':step,'epoch_seconds':time.monotonic()-epoch_started,
             **{'train_epoch/'+k:v/epoch_weight for k,v in epoch_sums.items()},
             **{'validation/'+k:v for k,v in metrics.items()}})
        save_checkpoint(args.output_dir/f'epoch{epoch}',model,tokenizer,head,snapshot(0))
        removed=prune_optimizer_states(args.output_dir,args.keep_optimizer_checkpoints)
        if removed:
            log({'epoch':epoch,'step':step,'pruned_optimizer_states':removed})
    if wandb_run: wandb_run.finish()


if __name__=='__main__': main()
