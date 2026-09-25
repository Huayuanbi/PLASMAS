"""Prompt, serialization, and weighted-pair utilities for topology DPO."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import random

from .data import PairwiseRewardDataset, RewardPair, TopologyExample


DPO_SYSTEM_PROMPT = """You design an efficient directed acyclic communication graph for a team of
mathematical agents. Infer the intrinsic difficulty of the problem, then dynamically choose the
smallest team that is still likely to solve it reliably. Easy problems should avoid unnecessary
agents and edges. Hard, fragile, or collaboration-intensive problems should use additional
planning, independent solving, and verification when those roles materially improve reliability.

Do not solve the mathematics problem. Return exactly one JSON object and no markdown or prose:
{
  "selected_agents": ["agent_id", ...],
  "edges": [["source_agent_id", "target_agent_id"], ...]
}

Rules:
1. The finalizer must always be selected.
2. Use only agent IDs listed by the user, with no duplicates.
3. Every edge endpoint must be selected; no self-edge or duplicate edge is allowed.
4. The graph must be a DAG. An edge source -> target means the source sends its output to target.
5. Every selected non-finalizer agent must have a directed path to the finalizer.
6. A one-node graph containing only the finalizer is allowed for genuinely easy problems.
7. Base the decision on the problem and supplied role briefs, not on a fixed template.
"""

MMLU_PRO_SYSTEM_PROMPT = """You design an efficient directed acyclic communication graph for a team of
subject-knowledge and multiple-choice reasoning agents. Infer the difficulty and the kinds of
knowledge needed from the question and options, then choose the smallest team likely to answer
reliably. Easy questions should avoid unnecessary agents and edges. Use additional terminology,
disciplinary knowledge, case analysis, distractor auditing, and evidence calibration only when
the supplied roles are likely to improve reliability.

Do not answer the multiple-choice question. Return exactly one JSON object and no markdown or prose:
{
  "selected_agents": ["agent_id", ...],
  "edges": [["source_agent_id", "target_agent_id"], ...]
}

Rules:
1. The finalizer must always be selected.
2. Use only agent IDs listed by the user, with no duplicates.
3. Every edge endpoint must be selected; no self-edge or duplicate edge is allowed.
4. The graph must be a DAG. An edge source -> target means the source sends its output to target.
5. Every selected non-finalizer agent must have a directed path to the finalizer.
6. A one-node graph containing only the finalizer is allowed for genuinely easy questions.
7. Base the decision on the question, options, and supplied role briefs, not on a fixed template.
"""


def topology_messages(task: str, nodes: list[dict], task_family: str = "math") -> list[dict]:
    """Shared training/generation prompt; never accepts answers or difficulty labels."""
    if task_family not in {"math", "mmlu_pro"}:
        raise ValueError(f"unsupported task family: {task_family}")
    roles = "\n\n".join(
        f"ID: {node['id']}\nROLE: {node['role']}\nROLE BRIEF:\n{node['role_brief']}"
        for node in nodes
    )
    header = "MATHEMATICS PROBLEM" if task_family == "math" else "MULTIPLE-CHOICE QUESTION"
    return [
        {"role": "system", "content": DPO_SYSTEM_PROMPT if task_family == "math" else MMLU_PRO_SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}:\n{task}\n\nAVAILABLE AGENTS:\n{roles}\n\nSelect the topology now."},
    ]


@dataclass(frozen=True)
class TopologyDPOExample:
    prompt: str
    chosen: str
    rejected: str
    weight: float
    category: str
    ranking_gap: float
    group_index: int


def topology_json(example: TopologyExample) -> str:
    """Serialize one labeled topology to the same compact JSON used for DPO."""

    node_ids = [str(node.id) for node in example.nodes]
    selected = [
        node_id
        for node_id, pruned in zip(node_ids, example.prune_mask.tolist())
        if float(pruned) == 0.0
    ]
    selected_set = set(selected)
    edges = [
        [node_ids[source], node_ids[target]]
        for source in range(example.num_nodes)
        for target in range(example.num_nodes)
        if float(example.edge_weight[source, target]) != 0.0
        and node_ids[source] in selected_set
        and node_ids[target] in selected_set
    ]
    return json.dumps(
        {"selected_agents": selected, "edges": edges},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def topology_prompt(example: TopologyExample, tokenizer, task_family: str = "math") -> str:
    """Render the Qwen chat prefix without an assistant completion."""

    return tokenizer.apply_chat_template(
        topology_messages(example.task, [dict(id=n.id, role=n.role, role_brief=n.role_brief)
                                        for n in example.nodes], task_family),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _sample_category_pairs(
    pairs: list[RewardPair], limit: int, rng: random.Random
) -> list[RewardPair]:
    if limit <= 0 or len(pairs) <= limit:
        return pairs
    return rng.sample(pairs, limit)


def build_dpo_examples(
    dataset: PairwiseRewardDataset,
    tokenizer,
    *,
    group_indices: set[int] | None = None,
    pairs_per_category: int = 4,
    ranking_gap_unit: float = 0.2,
    quality_pair_weight: float = 1.0,
    cost_pair_weight: float = 0.25,
    tradeoff_pair_weight: float = 0.5,
    chosen_mode: str = "pairwise",
    seed: int = 7,
) -> list[TopologyDPOExample]:
    """Flatten query groups while preserving grouped pair-loss weighting.

    Within a query and category, the existing grouped loss averages pair terms.
    Accordingly each selected DPO pair receives ``category_weight / count`` before
    applying its ranking-gap multiplier. A seeded per-category cap keeps repeated
    1,300-token prompts computationally manageable.
    """

    if pairs_per_category < 0:
        raise ValueError("pairs_per_category must be non-negative")
    if ranking_gap_unit <= 0:
        raise ValueError("ranking_gap_unit must be positive")
    if chosen_mode not in {"pairwise", "fit"}:
        raise ValueError("chosen_mode must be 'pairwise' or 'fit'")
    category_weights = {
        "quality": quality_pair_weight,
        "cost": cost_pair_weight,
        "tradeoff": tradeoff_pair_weight,
    }
    if any(weight < 0 for weight in category_weights.values()):
        raise ValueError("pair category weights must be non-negative")

    examples = []
    for group_index, group in enumerate(dataset.pair_groups):
        if group_indices is not None and group_index not in group_indices:
            continue
        prompt = topology_prompt(group.fit_candidate, tokenizer)
        by_category: dict[str, list[RewardPair]] = {
            "quality": [],
            "cost": [],
            "tradeoff": [],
        }
        for pair in group.pairs:
            if chosen_mode == "fit" and pair.preferred is not group.fit_candidate:
                continue
            by_category[pair.category].append(pair)
        rng = random.Random(seed + group_index)
        for category, category_pairs in by_category.items():
            selected_pairs = _sample_category_pairs(
                category_pairs, pairs_per_category, rng
            )
            if not selected_pairs or category_weights[category] == 0:
                continue
            category_count = len(selected_pairs)
            for pair in selected_pairs:
                chosen = topology_json(pair.preferred)
                rejected = topology_json(pair.rejected)
                if chosen == rejected:
                    continue
                weight = (
                    category_weights[category]
                    * (pair.ranking_gap / ranking_gap_unit)
                    / category_count
                )
                examples.append(
                    TopologyDPOExample(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        weight=weight,
                        category=category,
                        ranking_gap=pair.ranking_gap,
                        group_index=group_index,
                    )
                )
    return examples


def summarize_dpo_examples(examples: list[TopologyDPOExample]) -> str:
    categories = Counter(example.category for example in examples)
    groups = len({example.group_index for example in examples})
    mean_weight = (
        sum(example.weight for example in examples) / len(examples) if examples else 0.0
    )
    return (
        f"groups={groups} pairs={len(examples)} "
        + " ".join(f"{name}={categories[name]}" for name in ("quality", "cost", "tradeoff"))
        + f" mean_weight={mean_weight:.6f}"
    )


__all__ = [
    "DPO_SYSTEM_PROMPT",
    "TopologyDPOExample",
    "build_dpo_examples",
    "summarize_dpo_examples",
    "topology_json",
    "topology_prompt",
]
