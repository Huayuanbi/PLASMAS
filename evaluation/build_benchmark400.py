from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "math" / "test_anchors_5_delta_levels.json"
DEFAULT_OUTPUT = ROOT / "data" / "math" / "benchmark400.json"
CATEGORY_QUOTAS = {
    "Algebra": 15,
    "Prealgebra": 15,
    "Counting & Probability": 14,
    "Geometry": 14,
    "Intermediate Algebra": 14,
    "Number Theory": 14,
    "Precalculus": 14,
}
GROUP_ORDER = (
    "easy",
    "medium",
    "collaboration_required",
    "hard_unsolved",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a category-balanced 400-question collaboration-demand benchmark."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difficulty_group(record: dict) -> str | None:
    finalizer = float(record["finalizer_accuracy_avg5"])
    expert = float(record["expert_accuracy_avg5"])
    delta = float(record["delta_accuracy"])
    if math.isclose(finalizer, 1.0, abs_tol=1e-12):
        return "easy"
    if delta > 0.4 + 1e-12:
        return "collaboration_required"
    if finalizer <= 0.2 + 1e-12 and expert <= 0.2 + 1e-12:
        return "hard_unsolved"
    if 0.4 - 1e-12 <= finalizer <= 0.8 + 1e-12 and abs(delta) <= 0.2 + 1e-12:
        return "medium"
    return None


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    records = source["records"]
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    available_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        group = difficulty_group(record)
        if group is None:
            continue
        category = record["source_metadata"]["type"]
        pools[group][category].append(record)
        available_counts[group][category] += 1

    rng = random.Random(args.seed)
    selected = []
    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for group in GROUP_ORDER:
        for category, quota in CATEGORY_QUOTAS.items():
            candidates = sorted(
                pools[group][category], key=lambda record: int(record["source_index"])
            )
            if len(candidates) < quota:
                raise ValueError(
                    f"group={group} category={category}: need {quota}, have {len(candidates)}"
                )
            chosen = rng.sample(candidates, quota)
            chosen.sort(key=lambda record: int(record["source_index"]))
            for record in chosen:
                selected.append(
                    {
                        "benchmark_index": len(selected),
                        "difficulty_group": group,
                        **record,
                    }
                )
                selected_counts[group][category] += 1

    output = {
        "metadata": {
            "protocol": "query_topology_difficulty_v1",
            "created_date": "2026-09-01",
            "source": str(args.input),
            "source_sha256": sha256(args.input),
            "seed": args.seed,
            "questions_per_group": 100,
            "groups": {
                "easy": "finalizer_accuracy_avg5 = 1.0",
                "medium": "0.4 <= finalizer_accuracy_avg5 <= 0.8 and |delta_accuracy| <= 0.2",
                "collaboration_required": "delta_accuracy > 0.4",
                "hard_unsolved": "finalizer_accuracy_avg5 <= 0.2 and expert_accuracy_avg5 <= 0.2",
            },
            "category_quotas": CATEGORY_QUOTAS,
            "available_counts": {
                group: dict(available_counts[group]) for group in GROUP_ORDER
            },
            "selected_counts": {
                group: dict(selected_counts[group]) for group in GROUP_ORDER
            },
        },
        "records": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
