from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_MATH_TRAIN_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "math"
    / "math_train_1000x12_avg5.json"
)
DEFAULT_JOINT_TRAIN_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "joint"
    / "math_mmlu_train_2000x12_avg5.json"
)


@dataclass(frozen=True)
class NodeSpec:
    id: str | int
    role: str
    role_brief: str


@dataclass(frozen=True)
class TopologyExample:
    task: str
    nodes: tuple[NodeSpec, ...]
    prune_mask: torch.Tensor
    edge_weight: torch.Tensor
    edge_token_cost: torch.Tensor
    edge_time_cost: torch.Tensor
    total_input_tokens: float
    total_output_tokens: float
    reward: float | None
    allowed_edge_mask: torch.Tensor | None = None
    pair_group: int | None = None
    rollout_count: int | None = None
    correct_count: int | None = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_token_cost(self) -> float:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def has_rollout_counts(self) -> bool:
        """Whether the reward carries the Bernoulli counts behind it."""
        return (
            self.rollout_count is not None
            and self.correct_count is not None
            and self.rollout_count > 0
        )


def reward_difference_significance(
    left: TopologyExample, right: TopologyExample
) -> float | None:
    """Two-proportion z statistic for ``left.reward`` versus ``right.reward``.

    Rewards are means of Bernoulli rollouts, so a fixed reward gap is not a
    calibrated notion of "these two graphs differ". Under the null that both
    graphs share one success probability, the pooled standard error of the
    difference is ``sqrt(p (1 - p) (1/n1 + 1/n2))``; the returned value is the
    observed gap in units of that standard error.

    Returns ``None`` when either example lacks rollout counts, so callers can
    fall back to a fixed reward-gap rule on datasets without rollout metadata.
    A zero pooled variance can only arise when every rollout of both graphs
    agrees, in which case the gap is exactly zero and ``0.0`` is returned.
    """
    if not (left.has_rollout_counts and right.has_rollout_counts):
        return None
    assert left.rollout_count is not None and left.correct_count is not None
    assert right.rollout_count is not None and right.correct_count is not None
    left_n, right_n = left.rollout_count, right.rollout_count
    pooled = (left.correct_count + right.correct_count) / (left_n + right_n)
    variance = pooled * (1.0 - pooled) * (1.0 / left_n + 1.0 / right_n)
    if variance <= 0.0:
        return 0.0
    gap = abs(left.correct_count / left_n - right.correct_count / right_n)
    return gap / math.sqrt(variance)


@dataclass(frozen=True)
class RewardPair:
    """Two candidate graphs for the same task, ordered by scalar utility."""

    preferred: TopologyExample
    rejected: TopologyExample
    preferred_utility: float
    rejected_utility: float
    ranking_gap: float
    category: str

    @property
    def reward_gap(self) -> float:
        assert self.preferred.reward is not None
        assert self.rejected.reward is not None
        return self.preferred.reward - self.rejected.reward

    @property
    def utility_gap(self) -> float:
        return self.preferred_utility - self.rejected_utility


@dataclass(frozen=True)
class RewardPairGroup:
    """All candidate graphs and pairwise preferences for one query."""

    candidates: tuple[TopologyExample, ...]
    pairs: tuple[RewardPair, ...]
    fit_candidate: TopologyExample

    @property
    def task(self) -> str:
        return self.fit_candidate.task

    @property
    def nodes(self) -> tuple[NodeSpec, ...]:
        return self.fit_candidate.nodes


class AGPJsonDataset(Dataset[TopologyExample]):
    def __init__(self, path: str | Path, max_records: int | None = None) -> None:
        self.path = Path(path)
        self._node_pool_cache: dict[Path, list[dict]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if max_records is not None:
            records = records[:max_records]
        expanded_records = [
            candidate
            for group_index, record in enumerate(records)
            for candidate in self._expand_record(record, group_index)
        ]
        self.examples = [self._parse(record) for record in expanded_records]

    def _load_node_pool(self, reference: str) -> list[dict]:
        """Load a node pool referenced relative to the dataset JSON file."""
        pool_path = Path(reference)
        if not pool_path.is_absolute():
            pool_path = self.path.parent / pool_path
        pool_path = pool_path.resolve()

        if pool_path not in self._node_pool_cache:
            try:
                with pool_path.open("r", encoding="utf-8") as handle:
                    pool = json.load(handle)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"node_pool file not found: {reference!r} "
                    f"(resolved to {pool_path})"
                ) from exc
            if not isinstance(pool, dict) or not isinstance(pool.get("nodes"), list):
                raise ValueError(
                    f"node_pool {reference!r} must be an object containing a nodes array"
                )
            if not pool["nodes"]:
                raise ValueError(f"node_pool {reference!r} must not be empty")
            self._node_pool_cache[pool_path] = pool["nodes"]
        return self._node_pool_cache[pool_path]

    def _expand_record(self, record: dict, group_index: int = 0) -> list[dict]:
        """Flatten a grouped task's candidate graphs; accept legacy flat records."""
        if "graphs" not in record:
            return [record]

        graphs = record["graphs"]
        if not isinstance(graphs, list) or not graphs:
            raise ValueError("graphs must be a non-empty list")
        if "task" not in record:
            raise ValueError("grouped records require a top-level task")
        if "nodes" in record and "node_pool" in record:
            raise ValueError("use either nodes or node_pool, not both")
        if "nodes" in record:
            nodes = record["nodes"]
        elif isinstance(record.get("node_pool"), str) and record["node_pool"]:
            nodes = self._load_node_pool(record["node_pool"])
        else:
            raise ValueError("grouped records require top-level nodes or node_pool")

        expanded = []
        for index, graph in enumerate(graphs):
            if not isinstance(graph, dict):
                raise ValueError(f"graphs[{index}] must be an object")
            if any(
                key in graph
                for key in (
                    "task",
                    "nodes",
                    "node_pool",
                    "graphs",
                    "allowed_edge_mask",
                )
            ):
                raise ValueError(
                    f"graphs[{index}] must not override task, nodes, node_pool, "
                    "allowed_edge_mask, or graphs"
                )
            expanded.append(
                {
                    "task": record["task"],
                    "nodes": nodes,
                    "allowed_edge_mask": record.get("allowed_edge_mask"),
                    **graph,
                    "_pair_group": group_index,
                }
            )
        return expanded

    @staticmethod
    def _parse(record: dict) -> TopologyExample:
        prune_mask = torch.tensor(record["mask"], dtype=torch.float32)
        n = prune_mask.numel()

        def parse_square_matrix(
            name: str, default: torch.Tensor | None = None
        ) -> torch.Tensor:
            raw_matrix = record.get(name)
            if raw_matrix is None:
                if default is None:
                    raise ValueError(f"missing required matrix: {name}")
                matrix = default.clone()
            else:
                matrix = torch.as_tensor(raw_matrix, dtype=torch.float32)
            if matrix.shape != (n, n):
                raise ValueError(
                    f"{name} must have shape ({n}, {n}), got {tuple(matrix.shape)}"
                )
            if not torch.isfinite(matrix).all():
                raise ValueError(f"{name} must contain only finite floating-point values")
            return matrix

        edge_weight = parse_square_matrix("edge_weight")
        edge_token_cost = parse_square_matrix(
            "edge_token_cost", torch.zeros_like(edge_weight)
        )
        edge_time_cost = parse_square_matrix(
            "edge_time_cost", torch.zeros_like(edge_weight)
        )

        raw_allowed_edge_mask = record.get("allowed_edge_mask")
        if raw_allowed_edge_mask is None:
            allowed_edge_mask = None
        else:
            allowed_matrix = torch.as_tensor(raw_allowed_edge_mask, dtype=torch.float32)
            if allowed_matrix.shape != (n, n):
                raise ValueError(
                    "allowed_edge_mask must have shape "
                    f"({n}, {n}), got {tuple(allowed_matrix.shape)}"
                )
            if not torch.isfinite(allowed_matrix).all() or not torch.all(
                (allowed_matrix == 0) | (allowed_matrix == 1)
            ):
                raise ValueError("allowed_edge_mask must contain only 0 or 1")
            allowed_edge_mask = allowed_matrix.bool()
            if allowed_edge_mask.diagonal().any():
                raise ValueError("allowed_edge_mask must disable self-edges")
            forbidden = ~allowed_edge_mask
            for name, matrix in (
                ("edge_weight", edge_weight),
                ("edge_token_cost", edge_token_cost),
                ("edge_time_cost", edge_time_cost),
            ):
                if torch.any(matrix[forbidden] != 0):
                    raise ValueError(f"{name} must be zero on forbidden edges")

        def parse_non_negative_float(name: str, default: float) -> float:
            raw_value = record.get(name, default)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            return value

        fallback_token_cost = float(edge_token_cost.sum())
        total_input_tokens = parse_non_negative_float(
            "total_input_tokens", fallback_token_cost
        )
        total_output_tokens = parse_non_negative_float("total_output_tokens", 0.0)

        raw_reward = record.get("reward")
        if raw_reward is None:
            reward = None
        elif isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
            raise ValueError("reward must be a finite number or null")
        else:
            reward = float(raw_reward)
            if not math.isfinite(reward):
                raise ValueError("reward must be a finite number or null")

        def parse_count(name: str) -> int | None:
            raw_count = record.get(name)
            if raw_count is None:
                return None
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise ValueError(f"{name} must be a non-negative integer or null")
            if raw_count < 0:
                raise ValueError(f"{name} must be a non-negative integer or null")
            return raw_count

        rollout_count = parse_count("rollout_count")
        correct_count = parse_count("correct_count")
        if rollout_count is not None and correct_count is not None:
            if correct_count > rollout_count:
                raise ValueError("correct_count must not exceed rollout_count")

        raw_nodes = record.get("nodes")
        if raw_nodes is None:
            nodes = tuple(
                NodeSpec(
                    id=i,
                    role=f"node_{i}",
                    role_brief=f"Generic agent at position {i}.",
                )
                for i in range(n)
            )
        else:
            if len(raw_nodes) != n:
                raise ValueError(f"nodes must contain {n} entries, got {len(raw_nodes)}")
            parsed_nodes = []
            ids: set[str] = set()
            for index, raw_node in enumerate(raw_nodes):
                node_id = raw_node.get("id", index)
                normalized_id = str(node_id)
                if normalized_id in ids:
                    raise ValueError(f"node ids must be unique, duplicate: {node_id!r}")
                ids.add(normalized_id)
                role = str(raw_node.get("role", f"node_{index}"))
                role_brief = str(raw_node.get("role_brief", role)).strip()
                if not role_brief:
                    raise ValueError(f"nodes[{index}].role_brief must not be empty")
                parsed_nodes.append(
                    NodeSpec(
                        id=node_id,
                        role=role,
                        role_brief=role_brief,
                    )
                )
            nodes = tuple(parsed_nodes)

        return TopologyExample(
            task=record["task"],
            nodes=nodes,
            prune_mask=prune_mask,
            edge_weight=edge_weight,
            edge_token_cost=edge_token_cost,
            edge_time_cost=edge_time_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            reward=reward,
            allowed_edge_mask=allowed_edge_mask,
            pair_group=record.get("_pair_group"),
            rollout_count=rollout_count,
            correct_count=correct_count,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TopologyExample:
        return self.examples[index]


class PairwiseRewardDataset(Dataset[RewardPair]):
    """Create accuracy-first pairs with cost tradeoffs only for close rewards.

    Two behaviours are configurable because both were measured to matter:

    ``reward_significance_z``
        How a pair decides whether reward or cost sets the ordering. The default
        ``0.0`` keeps the legacy fixed ``cost_tradeoff_reward_gap`` threshold. A
        positive value instead requires the reward gap to reach that many pooled
        standard errors (see :func:`reward_difference_significance`), which makes
        the threshold adapt to the rollout budget and to where on ``[0, 1]`` the
        two rewards sit. Ignored for examples without rollout counts.

    ``fit_cost_weight``
        How the ``preferred_fit`` anchor is chosen. The default ``None`` keeps
        the legacy rule -- minimum token cost among the exact argmax-reward set
        -- which is equivalent to an *infinite* cost weight inside that set and
        therefore ignores ``token_cost_weight`` entirely. Passing a float
        selects ``argmax(reward - fit_cost_weight * cost / token_cost_scale)``
        instead, so the anchor responds to the cost preference like the pairs do.
        ``fit_cost_weight=0.0`` gives an accuracy-first anchor.

        This matters because the argmax-reward set is large (median 8 of 12
        candidates on the MATH data), so which member is cheapest is decided by
        rollout noise: the legacy anchor reproduces across disjoint rollout
        halves only 58% of the time and averages 2.27 active nodes, while
        ``fit_cost_weight=0.0`` reproduces 73% of the time and averages 5.43.
    """

    def __init__(
        self,
        dataset: AGPJsonDataset,
        min_utility_gap: float = 0.0,
        token_cost_weight: float = 0.02,
        token_cost_scale: float = 1000.0,
        cost_tradeoff_reward_gap: float = 0.2,
        reward_significance_z: float = 0.0,
        fit_cost_weight: float | None = None,
    ) -> None:
        if min_utility_gap < 0:
            raise ValueError("min_utility_gap must be non-negative")
        if token_cost_weight < 0:
            raise ValueError("token_cost_weight must be non-negative")
        if token_cost_scale <= 0:
            raise ValueError("token_cost_scale must be positive")
        if cost_tradeoff_reward_gap < 0:
            raise ValueError("cost_tradeoff_reward_gap must be non-negative")
        if reward_significance_z < 0:
            raise ValueError("reward_significance_z must be non-negative")
        if fit_cost_weight is not None and fit_cost_weight < 0:
            raise ValueError("fit_cost_weight must be non-negative")

        def utility(example: TopologyExample) -> float:
            assert example.reward is not None
            normalized_cost = example.total_token_cost / token_cost_scale
            return example.reward - token_cost_weight * normalized_cost

        def fit_utility(example: TopologyExample) -> float:
            assert example.reward is not None and fit_cost_weight is not None
            normalized_cost = example.total_token_cost / token_cost_scale
            return example.reward - fit_cost_weight * normalized_cost

        def reward_decides(left: TopologyExample, right: TopologyExample) -> bool:
            """Whether the reward gap, not cost, should order this pair."""
            assert left.reward is not None and right.reward is not None
            if reward_significance_z > 0:
                significance = reward_difference_significance(left, right)
                if significance is not None:
                    return significance >= reward_significance_z
            return abs(left.reward - right.reward) > cost_tradeoff_reward_gap

        groups: dict[object, list[TopologyExample]] = {}
        for example in dataset.examples:
            if example.reward is not None:
                key: object = (
                    ("grouped", example.pair_group)
                    if example.pair_group is not None
                    else ("legacy", example.task, example.nodes)
                )
                groups.setdefault(key, []).append(example)

        pairs: list[RewardPair] = []
        pair_groups: list[RewardPairGroup] = []
        for candidates in groups.values():
            group_pairs: list[RewardPair] = []
            for left, right in itertools.combinations(candidates, 2):
                left_utility = utility(left)
                right_utility = utility(right)
                assert left.reward is not None and right.reward is not None
                reward_gap = abs(left.reward - right.reward)

                if reward_decides(left, right):
                    left_is_preferred = left.reward > right.reward
                    ranking_gap = reward_gap
                else:
                    utility_gap = abs(left_utility - right_utility)
                    if utility_gap == 0:
                        continue
                    left_is_preferred = left_utility > right_utility
                    ranking_gap = utility_gap

                if ranking_gap <= min_utility_gap:
                    continue
                if left_is_preferred:
                    preferred, rejected = left, right
                    preferred_utility, rejected_utility = left_utility, right_utility
                else:
                    preferred, rejected = right, left
                    preferred_utility, rejected_utility = right_utility, left_utility
                assert preferred.reward is not None and rejected.reward is not None
                if preferred.reward == rejected.reward:
                    category = "cost"
                elif preferred.reward < rejected.reward:
                    category = "tradeoff"
                else:
                    category = "quality"
                pair = RewardPair(
                    preferred=preferred,
                    rejected=rejected,
                    preferred_utility=preferred_utility,
                    rejected_utility=rejected_utility,
                    ranking_gap=ranking_gap,
                    category=category,
                )
                pairs.append(pair)
                group_pairs.append(pair)

            if candidates:
                if fit_cost_weight is None:
                    # Legacy: minimum token cost among the exact argmax-reward set.
                    max_reward = max(
                        candidate.reward
                        for candidate in candidates
                        if candidate.reward is not None
                    )
                    fit_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate.reward is not None
                        and math.isclose(candidate.reward, max_reward, abs_tol=1e-12)
                    ]
                    fit_candidate = min(
                        fit_candidates,
                        key=lambda candidate: candidate.total_token_cost,
                    )
                else:
                    # One explicit utility, so the anchor follows fit_cost_weight.
                    fit_candidate = max(
                        (
                            candidate
                            for candidate in candidates
                            if candidate.reward is not None
                        ),
                        key=fit_utility,
                    )
                pair_groups.append(
                    RewardPairGroup(
                        candidates=tuple(candidates),
                        pairs=tuple(group_pairs),
                        fit_candidate=fit_candidate,
                    )
                )
        self.pairs = pairs
        self.pair_groups = pair_groups

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> RewardPair:
        return self.pairs[index]


def summarize_pair_supervision(dataset: PairwiseRewardDataset) -> str:
    """One log line describing the pair mix and the preferred_fit anchor.

    The anchor's mean active-node count is worth watching: it is the direct
    target of the ``preferred_fit`` term, and the free optimum of the training
    loss lands within about 0.2 nodes of it.
    """
    categories = {"quality": 0, "cost": 0, "tradeoff": 0}
    for pair in dataset.pairs:
        categories[pair.category] += 1
    counts = " ".join(f"{name}={count}" for name, count in categories.items())
    groups = dataset.pair_groups
    if not groups:
        return f"pairs=0 {counts}"
    anchor_nodes = [
        float((group.fit_candidate.prune_mask == 0).sum()) for group in groups
    ]
    anchor_edges = [
        float((group.fit_candidate.edge_weight != 0).sum()) for group in groups
    ]
    return (
        f"pairs={len(dataset.pairs)} {counts} "
        f"fit_anchor_nodes={sum(anchor_nodes) / len(anchor_nodes):.2f} "
        f"fit_anchor_edges={sum(anchor_edges) / len(anchor_edges):.2f}"
    )
