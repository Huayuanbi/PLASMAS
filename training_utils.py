from __future__ import annotations
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from plasmas.qwen_topology_dpo import TopologyDPOExample

class FlatPairDataset(Dataset):
    def __init__(self, examples: list[TopologyDPOExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[int, TopologyDPOExample]:
        return index, self.examples[index]

class DPOCollator:
    PAD_TO_MULTIPLE = 8

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[tuple[int, TopologyDPOExample]]) -> dict:
        indices, examples = zip(*rows)
        prompts = [example.prompt for example in examples]
        completions = [example.chosen for example in examples] + [
            example.rejected for example in examples
        ]
        doubled_prompts = prompts + prompts
        eos = self.tokenizer.eos_token or ""
        prompt_rows = self.tokenizer(doubled_prompts, add_special_tokens=False)[
            "input_ids"
        ]
        completion_rows = self.tokenizer(
            [completion + eos for completion in completions], add_special_tokens=False
        )["input_ids"]
        lengths = [
            len(prompt_ids) + len(completion_ids)
            for prompt_ids, completion_ids in zip(prompt_rows, completion_rows)
        ]
        if any(length > self.max_length for length in lengths):
            raise ValueError(
                f"prompt plus completion exceeds --max-length={self.max_length}; "
                f"longest row in this batch is {max(lengths)} tokens"
            )
        raw_max_length = max(lengths)
        max_length = (
            (raw_max_length + self.PAD_TO_MULTIPLE - 1) // self.PAD_TO_MULTIPLE
        ) * self.PAD_TO_MULTIPLE
        if max_length > self.max_length:
            raise ValueError(
                f"batch requires {max_length} tokens after padding to a multiple "
                f"of {self.PAD_TO_MULTIPLE}, exceeding --max-length={self.max_length}"
            )
        input_ids = torch.full(
            (len(lengths), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row_index, (prompt_ids, completion_ids) in enumerate(
            zip(prompt_rows, completion_rows)
        ):
            full_ids = prompt_ids + completion_ids
            input_ids[row_index, : len(full_ids)] = torch.tensor(full_ids)
            attention_mask[row_index, : len(full_ids)] = 1
            labels[
                row_index, len(prompt_ids) : len(full_ids)
            ] = torch.tensor(completion_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "indices": torch.tensor(indices, dtype=torch.long),
            "weights": torch.tensor(
                [example.weight for example in examples], dtype=torch.float32
            ),
            "categories": [example.category for example in examples],
        }

def completion_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum completion log-probabilities, optionally from sparse sequence logits.

    A causal logit at position ``i`` scores the label at ``i + 1``.  Qwen's
    ``logits_to_keep`` can therefore skip the very large prompt vocabulary
    projection while retaining exactly the positions needed by the DPO loss.
    """

    if logit_indices is None:
        shifted_logits = logits[:, :-1]
        shifted_labels = labels[:, 1:]
    else:
        if logits.size(1) != logit_indices.numel():
            raise ValueError("sparse logits and logit_indices have different lengths")
        shifted_logits = logits
        shifted_labels = labels.index_select(1, logit_indices + 1)
    valid = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = -F.cross_entropy(
        shifted_logits.transpose(1, 2), safe_labels, reduction="none"
    )
    return (token_logps.float() * valid).sum(dim=-1)

def dpo_losses(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_ratio = policy_chosen - policy_rejected
    reference_ratio = reference_chosen - reference_rejected
    logits = beta * (policy_ratio - reference_ratio)
    return -F.logsigmoid(logits), logits

def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
        "weights": batch["weights"].to(device),
    }

def _forward_completion_logps(model, batch: dict, rows: slice) -> torch.Tensor:
    input_ids = batch["input_ids"][rows]
    attention_mask = batch["attention_mask"][rows]
    labels = batch["labels"][rows]
    # Only completion labels contribute to DPO. Projecting every prompt hidden
    # state to Qwen's full vocabulary creates a multi-GB logits tensor for long
    # role briefs and was the source of the observed peak-memory OOM.
    valid_logit_columns = (labels[:, 1:] != -100).any(dim=0)
    logit_indices = valid_logit_columns.nonzero(as_tuple=False).flatten()
    if logit_indices.numel() == 0:
        raise ValueError("DPO batch contains no completion tokens")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=logit_indices,
    )
    logps = completion_logps(outputs.logits, labels, logit_indices)
    del outputs
    return logps

def policy_and_reference_logps(
    model,
    batch: dict,
    activation_offload_threshold: int = 1800,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_count = batch["weights"].numel()
    chosen_rows = slice(0, pair_count)
    rejected_rows = slice(pair_count, 2 * pair_count)

    # Compute the no-grad reference first so its temporary MLP tensors do not
    # overlap the live policy autograd graph.
    with torch.no_grad(), model.disable_adapter():
        reference = torch.cat(
            (
                _forward_completion_logps(model, batch, chosen_rows),
                _forward_completion_logps(model, batch, rejected_rows),
            )
        )

    # Do not put chosen and rejected in the same transformer forward. For rare
    # long prompts, also move tensors saved for backward to CPU. This keeps the
    # exact objective and avoids a deterministic activation spike near 2.3k
    # tokens without slowing down the common shorter examples.
    should_offload = (
        batch["input_ids"].is_cuda
        and activation_offload_threshold > 0
        and batch["input_ids"].size(1) >= activation_offload_threshold
    )
    policy_context = (
        # The collator pads the sequence dimension to a multiple of eight, so
        # SDPA tensors restored by the official hook retain aligned strides.
        # A custom pack/unpack hook caused illegal memory accesses and NaN
        # gradients for some long samples.
        torch.autograd.graph.save_on_cpu(pin_memory=False, device_type="cuda")
        if should_offload
        else nullcontext()
    )
    with policy_context:
        policy_chosen = _forward_completion_logps(model, batch, chosen_rows)
        policy_rejected = _forward_completion_logps(model, batch, rejected_rows)
        policy = torch.cat((policy_chosen, policy_rejected))
    return policy, reference
