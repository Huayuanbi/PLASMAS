"""Generate topologies, summarize fresh executions, and select on validation only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(source, split):
    rows = read(source)
    if isinstance(rows, dict):
        rows = rows['records']
    # Raw MMLU-Pro local splits identify questions in source_metadata and do
    # not carry the execution pipeline's ordinal source_index yet.
    rows = [dict(row, source_index=row.get('source_index', index)) for index,row in enumerate(rows)]
    if split == 'validation':
        rows = [r for r in rows if r.get('split') == 'validation']
        if len(rows) != 100:
            raise ValueError('validation requires all 100 original SFT validation questions')
        for row in rows:
            meta = row['source_metadata']
            if meta.get('local_split', meta.get('split')) != 'train':
                raise ValueError('validation must come from the training corpus')
    if not rows or len({r['task'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate evaluation tasks')
    if len({r['source_index'] for r in rows}) != len(rows):
        raise ValueError('duplicate source indices')
    return rows


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from plasmas.qwen_topology_dpo import topology_messages
    from evaluation.topology_io import extract_json, validate_topology
    from data_generation.dpo_refresh import graph

    rows = load_records(args.source, args.split)
    nodes = read(args.roles)['nodes']
    seeds = list(range(142, 147)) if args.split == 'validation' else [42]
    identity = dict(name=args.name, epoch=args.epoch, split=args.split,
                    source_sha256=digest(args.source), roles_sha256=digest(args.roles),
                    model=args.model, task_family=args.task_family, mas_seeds=seeds,
                    sft_sha256=digest(args.sft_adapter/'adapter_model.safetensors'),
                    checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                    dpo_sha256=digest(args.checkpoint/'adapter_model.safetensors') if args.checkpoint else None,
                    topology_temperature=0, topology_max_tokens=256, topology_seed=7,
                    mas_temperature=.7, mas_max_tokens=2048, corrective_attempts=1)
    out = args.output
    if (out/'identity.json').exists() and read(out/'identity.json') != identity:
        raise ValueError('evaluation identity changed; choose a new output directory')
    write(out/'identity.json', identity)
    saved = read(out/'topologies.json') if (out/'topologies.json').exists() else []
    expected = {r['source_index']: r for r in rows}
    done = {r['source_index'] for r in saved}
    if len(done) != len(saved) or not done <= expected.keys() or any(
            r['task'] != expected[r['source_index']]['task'] for r in saved):
        raise ValueError('invalid generation resume file')
    pending = [r for r in rows if r['source_index'] not in done]
    if pending:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device)
        model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
        if args.checkpoint:
            if not (args.checkpoint/'COMPLETE').is_file():
                raise ValueError('incomplete DPO checkpoint')
            model = PeftModel.from_pretrained(model, args.checkpoint).merge_and_unload()
        model.eval()
        for row in pending:
            torch.manual_seed(7 + int(row['source_index']))
            messages = topology_messages(row['task'], nodes, args.task_family)
            responses, input_tokens, output_tokens, topology, error = [], 0, 0, None, None
            for attempt in range(2):
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                        enable_thinking=False, return_tensors='pt').to(args.device)
                with torch.inference_mode():
                    seq = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            do_sample=False, max_new_tokens=256, use_cache=True,
                            pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)
                responses.append(answer)
                input_tokens += ids.shape[1]
                output_tokens += seq.shape[1] - ids.shape[1]
                try:
                    topology = validate_topology(extract_json(answer), nodes, require_difficulty=False)
                    error = None
                    break
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    error = str(exc)
                    messages.extend([{'role': 'assistant', 'content': answer}, {'role': 'user',
                        'content': f'Your response was invalid: {error}. Return a corrected JSON object only, following every graph rule.'}])
            result = {k: copy.deepcopy(v) for k,v in row.items() if k not in ('graphs', 'candidates', 'sft_generations')}
            result.update(node_pool=str(args.roles.resolve()), evaluator=args.task_family,
                          topology=topology, generation=dict(input_tokens=input_tokens,
                          output_tokens=output_tokens, responses=responses, error=error), graphs=[])
            if topology is not None:
                for seed in seeds:
                    result['graphs'].append({**graph(topology, nodes),
                        'id': f"q{row['source_index']}_s{seed}", 'sampling_seed': seed,
                        'generator': args.name})
            saved.append(result)
            write(out/'topologies.json', saved)
            print(f'{args.name}: {len(saved)}/{len(rows)}', flush=True)
    write(out/'candidates.json', sorted(saved, key=lambda r: r['source_index']))


def summarize(args):
    identity = read(args.output/'identity.json')
    rows = read(args.output/'scored.json')
    candidates = read(args.output/'candidates.json')
    expected = {r['source_index']: r for r in candidates}
    if len(rows) != len(expected) or {r['source_index'] for r in rows} != expected.keys():
        raise ValueError('incomplete evaluation')
    scores, costs, topology_costs = [], [], []
    for row in rows:
        original = expected[row['source_index']]
        if row['task'] != original['task'] or row['reference_answer'] != original['reference_answer']:
            raise ValueError('evaluation identity mismatch')
        topology_tokens = sum(original['generation'][k] for k in ('input_tokens', 'output_tokens'))
        topology_costs.append(topology_tokens)
        if original['topology'] is None:
            if row['graphs']:
                raise ValueError('invalid topology must execute no agents')
            scores.append(0.); costs.append(float(topology_tokens))
            continue
        runs = row['graphs']
        if sorted(g['sampling_seed'] for g in runs) != identity['mas_seeds']:
            raise ValueError('missing or repeated execution seeds')
        for g, target in zip(runs, original['graphs']):
            if g['mask'] != target['mask'] or g['edge_weight'] != target['edge_weight']:
                raise ValueError('executed topology changed')
            if g.get('execution_status') != 'completed' or g.get('accuracy') not in (0, 1):
                raise ValueError('incomplete execution; no report emitted')
            if g.get('prediction') is None and g['accuracy'] != 0:
                raise ValueError('unparsed prediction must count as wrong')
        scores.append(sum(g['accuracy'] for g in runs)/len(runs))
        costs.append(topology_tokens + sum(g['total_input_tokens']+g['total_output_tokens'] for g in runs)/len(runs))
    report = {**identity, 'questions': len(rows),
              'invalid_topologies': sum(r['topology'] is None for r in candidates),
              'accuracy': sum(scores)/len(rows), 'mean_total_tokens': sum(costs)/len(rows),
              'mean_topology_tokens': sum(topology_costs)/len(rows),
              'scored_sha256': digest(args.output/'scored.json')}
    write(args.output/'report.json', report)
    print(json.dumps(report, indent=2))


def choose(reports, tolerance=.01):
    if not reports or any(r['split'] != 'validation' or r['questions'] != 100 for r in reports):
        raise ValueError('checkpoint selection requires 100-question validation reports only')
    fields = ('source_sha256', 'roles_sha256', 'model', 'sft_sha256', 'task_family',
              'mas_seeds', 'mas_temperature', 'mas_max_tokens', 'topology_temperature',
              'topology_max_tokens', 'topology_seed', 'corrective_attempts')
    if len({json.dumps([r[k] for k in fields], sort_keys=True) for r in reports}) != 1:
        raise ValueError('incompatible validation protocols')
    valid = [r for r in reports if r['invalid_topologies'] == 0]
    trials = [r for r in valid if r['checkpoint'] is not None]
    if not trials:
        raise ValueError('no valid DPO checkpoint')
    best = max(r['accuracy'] for r in valid)
    eligible = [r for r in trials if r['accuracy'] >= best-tolerance-1e-9]
    selected = (min(eligible, key=lambda r:(r['mean_total_tokens'], -r['accuracy'], r['epoch']))
                if eligible else min(trials, key=lambda r:(-r['accuracy'], r['mean_total_tokens'], r['epoch'])))
    return dict(selected=selected['name'], checkpoint=selected['checkpoint'],
                accuracy_constraint_met=bool(eligible), best_validation_accuracy=best,
                accuracy_tolerance=tolerance, reports=reports)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['generate', 'summarize', 'select'])
    p.add_argument('--source', type=Path)
    p.add_argument('--roles', type=Path)
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--sft-adapter', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--task-family', choices=['math', 'mmlu_pro'], default='math')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--name', default='plasmas')
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reports', nargs='+', type=Path)
    args = p.parse_args()
    if args.stage == 'generate':
        generate(args)
    elif args.stage == 'summarize':
        summarize(args)
    else:
        write(args.output/'selection.json', choose([read(path) for path in args.reports]))


if __name__ == '__main__':
    main()
