from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from plasmas import AGPJsonDataset, PairwiseRewardDataset
from plasmas.qwen_topology_dpo import topology_json, topology_prompt


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    completion: str
    group_index: int


class FlatSFTDataset(Dataset):
    def __init__(self, examples: list[SFTExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[int, SFTExample]:
        return index, self.examples[index]


class SFTCollator:
    PAD_TO_MULTIPLE = 8

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[tuple[int, SFTExample]]) -> dict:
        indices, examples = zip(*rows)
        prompt_rows = self.tokenizer(
            [example.prompt for example in examples], add_special_tokens=False
        )["input_ids"]
        eos = self.tokenizer.eos_token or ""
        completion_rows = self.tokenizer(
            [example.completion + eos for example in examples], add_special_tokens=False
        )["input_ids"]
        lengths = [
            len(prompt) + len(completion)
            for prompt, completion in zip(prompt_rows, completion_rows)
        ]
        raw_length = max(lengths)
        padded_length = (
            (raw_length + self.PAD_TO_MULTIPLE - 1) // self.PAD_TO_MULTIPLE
        ) * self.PAD_TO_MULTIPLE
        if padded_length > self.max_length:
            raise ValueError(
                f"batch requires {padded_length} tokens but --max-length={self.max_length}"
            )
        input_ids = torch.full(
            (len(rows), padded_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row, (prompt, completion) in enumerate(zip(prompt_rows, completion_rows)):
            full = prompt + completion
            input_ids[row, : len(full)] = torch.tensor(full)
            attention_mask[row, : len(full)] = 1
            labels[row, len(prompt) : len(full)] = torch.tensor(completion)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "indices": torch.tensor(indices),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Completion-only LoRA SFT on FIT topology JSON")
    parser.add_argument("--data", type=Path, default=ROOT / "data/math/math_train_1000x12_avg5.json")
    parser.add_argument("--task-family", choices=("math", "mmlu_pro"), default="math")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--activation-offload-threshold", type=int, default=1800)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-data", type=Path, default=None,
                        help="Explicit disjoint FIT validation data; all --data records become training")
    parser.add_argument("--sampling-policy", choices=("uniform", "node-strata"), default="uniform",
                        help="node-strata gives equal expected mass to nonempty 1/2/3+ node strata")
    parser.add_argument("--save-epochs", default="",
                        help="Comma-separated epoch checkpoints to keep; empty saves every epoch")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--min-utility-gap", type=float, default=0.02)
    parser.add_argument("--token-cost-weight", type=float, default=0.02)
    parser.add_argument("--token-cost-scale", type=float, default=1000.0)
    parser.add_argument("--cost-tradeoff-reward-gap", type=float, default=0.2)
    parser.add_argument("--reward-significance-z", type=float, default=0.0)
    parser.add_argument("--fit-cost-weight", type=float, default=None)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare_examples(args, tokenizer) -> tuple[list[SFTExample], list[SFTExample]]:
    dataset = AGPJsonDataset(args.data)
    pair_dataset = PairwiseRewardDataset(
        dataset,
        min_utility_gap=args.min_utility_gap,
        token_cost_weight=args.token_cost_weight,
        token_cost_scale=args.token_cost_scale,
        cost_tradeoff_reward_gap=args.cost_tradeoff_reward_gap,
        reward_significance_z=args.reward_significance_z,
        fit_cost_weight=args.fit_cost_weight,
    )
    examples = [
        SFTExample(
            prompt=topology_prompt(group.fit_candidate, tokenizer, getattr(args, "task_family", "math")),
            completion=topology_json(group.fit_candidate),
            group_index=index,
        )
        for index, group in enumerate(pair_dataset.pair_groups)
    ]
    validation_path = getattr(args, "validation_data", None)
    if validation_path is not None:
        validation_dataset = AGPJsonDataset(validation_path)
        if {x.task for x in dataset} & {x.task for x in validation_dataset}:
            raise ValueError("explicit train/validation tasks overlap")
        # Reuse exactly the same FIT selection logic for explicit validation.
        from copy import copy
        validation_args = copy(args)
        validation_args.data = validation_path
        validation_args.validation_data = None
        validation_args.validation_fraction = 0.0
        validation, _ = prepare_examples(validation_args, tokenizer)
        if not examples or not validation:
            raise ValueError("empty explicit train or validation set")
        return examples, validation
    order = list(range(len(examples)))
    random.Random(args.seed).shuffle(order)
    validation_count = int(round(len(order) * args.validation_fraction))
    validation_indices = set(order[:validation_count])
    train = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return train, validation


def node_strata_weights(examples):
    from collections import Counter
    strata = [min(3, len(json.loads(e.completion)["selected_agents"])) for e in examples]
    if not strata or min(strata) < 1:
        raise ValueError("empty or invalid topology targets")
    counts = Counter(strata)
    return [1.0 / counts[s] for s in strata]


def sequence_summary(tokenizer, examples: list[SFTExample]) -> dict:
    eos = tokenizer.eos_token or ""
    prompt_lengths = tokenizer(
        [example.prompt for example in examples], add_special_tokens=False, return_length=True
    )["length"]
    completion_lengths = tokenizer(
        [example.completion + eos for example in examples],
        add_special_tokens=False,
        return_length=True,
    )["length"]
    lengths = [a + b for a, b in zip(prompt_lengths, completion_lengths)]
    return {
        "records": len(examples),
        "mean_tokens": float(np.mean(lengths)),
        "max_tokens": max(lengths),
        "mean_completion_tokens": float(np.mean(completion_lengths)),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def completion_loss(model, batch: dict, activation_offload_threshold: int) -> tuple[torch.Tensor, dict]:
    labels = batch["labels"]
    valid_columns = (labels[:, 1:] != -100).any(dim=0)
    logit_indices = valid_columns.nonzero(as_tuple=False).flatten()
    if logit_indices.numel() == 0:
        raise ValueError("SFT batch contains no completion tokens")
    should_offload = (
        batch["input_ids"].is_cuda
        and activation_offload_threshold > 0
        and batch["input_ids"].size(1) >= activation_offload_threshold
    )
    context = (
        torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda")
        if should_offload
        else nullcontext()
    )
    with context:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            logits_to_keep=logit_indices,
        )
        shifted_labels = labels.index_select(1, logit_indices + 1)
        valid = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~valid, 0)
        token_losses = F.cross_entropy(
            outputs.logits.transpose(1, 2), safe_labels, reduction="none"
        )
        token_count = valid.sum()
        loss = (token_losses * valid).sum() / token_count
        predictions = outputs.logits.detach().argmax(dim=-1)
        correct = ((predictions == safe_labels) & valid).sum()
    return loss, {
        "correct": int(correct),
        "tokens": int(token_count),
        "nll_sum": float((token_losses.detach() * valid).sum()),
    }


@torch.no_grad()
def evaluate(model, loader, device, offload_threshold: int) -> dict:
    model.eval()
    nll_sum = 0.0
    correct = 0
    tokens = 0
    for batch in loader:
        batch = move_batch(batch, device)
        _, metrics = completion_loss(model, batch, offload_threshold)
        nll_sum += metrics["nll_sum"]
        correct += metrics["correct"]
        tokens += metrics["tokens"]
    model.train()
    return {"loss": nll_sum / tokens, "token_accuracy": correct / tokens, "tokens": tokens}


def save_checkpoint(model, tokenizer, path: Path, args, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "training_state.json").write_text(
        json.dumps(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                **state,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples, validation_examples = prepare_examples(args, tokenizer)
    lengths = sequence_summary(tokenizer, train_examples + validation_examples)
    print(
        json.dumps(
            {
                "train_records": len(train_examples),
                "validation_records": len(validation_examples),
                "sequence_lengths": lengths,
                "sample_target": train_examples[0].completion,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if lengths["max_tokens"] > args.max_length:
        raise ValueError(
            f"dataset needs {lengths['max_tokens']} tokens, above max length {args.max_length}"
        )
    if args.dry_run:
        return

    from peft import LoraConfig, get_peft_model

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
            bias="none",
        ),
    )
    model.train()
    model.print_trainable_parameters()

    collator = SFTCollator(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (WeightedRandomSampler(node_strata_weights(train_examples),
               num_samples=len(train_examples), replacement=True, generator=generator)
               if args.sampling_policy == "node-strata" else None)
    train_loader = DataLoader(
        FlatSFTDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        FlatSFTDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda"
    )

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            mode=args.wandb_mode,
            config={
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "train_records": len(train_examples),
                "validation_records": len(validation_examples),
            },
            job_type="train",
        )
        wandb_run.define_metric("optimizer_step")
        wandb_run.define_metric("train/*", step_metric="optimizer_step")
        wandb_run.define_metric("validation/*", step_metric="optimizer_step")
        wandb_run.define_metric("memory/*", step_metric="optimizer_step")

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    global_step = 0
    best_validation_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        running_nll = 0.0
        running_correct = 0
        running_tokens = 0
        for micro_step, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = move_batch(batch, device)
            loss, metrics = completion_loss(model, batch, args.activation_offload_threshold)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite SFT loss at examples {batch['indices'].tolist()}"
                )
            remainder = len(train_loader) % args.gradient_accumulation_steps
            final_block_start = len(train_loader) - remainder + 1
            divisor = (
                remainder
                if remainder and micro_step >= final_block_start
                else args.gradient_accumulation_steps
            )
            (loss / divisor).backward()
            running_nll += metrics["nll_sum"]
            running_correct += metrics["correct"]
            running_tokens += metrics["tokens"]
            should_step = (
                micro_step % args.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, args.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % args.log_every == 0:
                    metrics_out = {
                        "train/loss": running_nll / running_tokens,
                        "train/token_accuracy": running_correct / running_tokens,
                        "train/grad_norm": float(grad_norm),
                        "train/epoch": epoch,
                        "memory/allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                        "memory/peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                    print(
                        f"epoch={epoch} optimizer_step={optimizer_step} "
                        f"loss={metrics_out['train/loss']:.6f} "
                        f"token_accuracy={metrics_out['train/token_accuracy']:.4f} "
                        f"elapsed={time.perf_counter()-started:.1f}s",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log({"optimizer_step": optimizer_step, **metrics_out})
                    running_nll = 0.0
                    running_correct = 0
                    running_tokens = 0

        validation = evaluate(
            model, validation_loader, device, args.activation_offload_threshold
        )
        improved = validation["loss"] < best_validation_loss
        best_validation_loss = min(best_validation_loss, validation["loss"])
        state = {
            "epoch": epoch,
            "epoch_complete": True,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "validation": validation,
            "best_validation_loss": best_validation_loss,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "epoch_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        save_epochs = {int(x) for x in args.save_epochs.split(",") if x.strip()}
        if not save_epochs or epoch in save_epochs:
            save_checkpoint(model, tokenizer, args.output_dir / f"epoch-{epoch}", args, state)
        if improved:
            save_checkpoint(model, tokenizer, args.output_dir / "best", args, state)
        print(
            f"epoch={epoch} validation_loss={validation['loss']:.6f} "
            f"validation_token_accuracy={validation['token_accuracy']:.4f} best={improved}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "optimizer_step": optimizer_step,
                    "validation/loss": validation["loss"],
                    "validation/token_accuracy": validation["token_accuracy"],
                    "validation/epoch": epoch,
                    "validation/best_loss": best_validation_loss,
                }
            )

    save_checkpoint(
        model,
        tokenizer,
        args.output_dir / "final",
        args,
        {
            "epoch": args.epochs,
            "epoch_complete": True,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "best_validation_loss": best_validation_loss,
        },
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
