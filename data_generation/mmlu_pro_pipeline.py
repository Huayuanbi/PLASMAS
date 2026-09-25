#!/usr/bin/env python3
"""MMLU-Pro download, knowledge-focused splits and 12-topology pipeline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path
import random
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
from data_generation import math_pipeline as common
from plasmas.topology_sampling import SampledTopology, validate_topology

REPOSITORY = 'TIGER-Lab/MMLU-Pro'
REVISION = 'b189ec765aa7ed75c8acfea42df31fdae71f97be'
KNOWLEDGE_CATEGORIES = {'biology', 'health', 'psychology', 'history', 'law', 'philosophy'}
ROLES = ('terminology_curator', 'disciplinary_memory', 'case_rule_specialist',
         'distractor_auditor', 'evidence_calibrator', 'finalizer')
# The role order is fixed. Every earlier active role may send to every later
# active role, while reverse edges and self-edges remain impossible.
ALLOWED = {(source, target) for source in range(6) for target in range(source + 1, 6)}
ANCHORS = [
    ('expert_anchor', {(0, 3), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4), (4, 5)}),
    ('terminology_contrast', {(0, 3), (3, 5)}),
    ('fact_triangulation', {(1, 4), (2, 4), (4, 5)}),
    ('independent_evidence', {(0, 5), (1, 5), (2, 5)}),
    ('distractor_calibration', {(1, 3), (3, 4), (4, 5)}),
    ('two_node', {(1, 5)}),
]


def category_in_profile(category, profile):
    if profile == 'knowledge':
        return category in KNOWLEDGE_CATEGORIES
    if profile == 'non_math':
        return category != 'math'
    if profile == 'full':
        return True
    raise ValueError(f'unknown profile: {profile}')


def resolve_pool(path):
    nodes, finalizer, roles = common.resolve_pool(path, ROLES)
    if len(nodes) != 6 or tuple(node['role'] for node in nodes) != ROLES or finalizer != 5:
        raise ValueError('MMLU-Pro requires the ordered six-role pool with finalizer last')
    return nodes, finalizer, roles


def topology(name, edges):
    active = {5} | {node for edge in edges for node in edge}
    graph = SampledTopology(name, tuple(0 if i in active else 1 for i in range(6)),
                            tuple(tuple(int((i, j) in edges) for j in range(6)) for i in range(6)),
                            tuple(sorted(active)))
    validate_topology(graph, 5)
    return graph


def role_anchors(num_nodes, finalizer, roles, *, fixed_order=None):
    return [topology(name, edges) for name, edges in ANCHORS]


@lru_cache(maxsize=1)
def legal_random_topologies():
    """Enumerate the fixed legal graph space once for all questions."""
    excluded = {
        topology('finalizer_only', set()).signature,
        *(topology(name, edges).signature for name, edges in ANCHORS),
    }
    legal = defaultdict(list)
    edges = sorted(ALLOWED)
    for bits in itertools.product((0, 1), repeat=len(edges)):
        selected = {edge for edge, bit in zip(edges, bits) if bit}
        try:
            graph = topology('role_random', selected)
        except ValueError:
            continue
        if graph.signature not in excluded:
            legal[len(graph.active_nodes)].append(graph)
    return tuple(tuple(legal[size]) for size in range(2, 7))


def candidate_suite(*, num_nodes, finalizer, role_indices, random_count=5, seed=42, fixed_order=None):
    if num_nodes != 6 or finalizer != 5 or random_count != 5:
        raise ValueError('MMLU-Pro suite requires six nodes and exactly five random DAGs')
    suite = [topology('finalizer_only', set()), *role_anchors(6, 5, role_indices)]
    rng = random.Random(seed)
    # One random graph at each active size 2..6, guaranteeing cost/size diversity.
    for choices in legal_random_topologies():
        suite.append(rng.choice(choices))
    return suite


def normalize(row, split, revision=REVISION):
    options = row['options']
    if not 2 <= len(options) <= 10 or any(not isinstance(x, str) for x in options):
        raise ValueError('expected 2..10 text options')
    index = row['answer_index']
    if not isinstance(index, int) or not 0 <= index < len(options) or row['answer'] != chr(65 + index):
        raise ValueError('inconsistent answer and answer_index')
    task = f"Subject: {row['category']}\nQuestion: {row['question']}\nOptions:\n"
    task += '\n'.join(f'{chr(65+i)}. {value}' for i, value in enumerate(options))
    task += '\nChoose the single best option. Start with FINAL_ANSWER: X (one option letter).'
    return {'task': task, 'reference_answer': row['answer'],
            'source_metadata': {'dataset': REPOSITORY, 'revision': revision,
                                'official_split': split, 'question_id': row['question_id'],
                                'category': row['category'], 'src': row.get('src'),
                                'option_count': len(options)}}


def download(args):
    raw = args.output_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': REPOSITORY, 'revision': REVISION, 'files': {}}
    for name in ('README.md', 'data/test-00000-of-00001.parquet',
                 'data/validation-00000-of-00001.parquet'):
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}'
        temporary = target.with_suffix(target.suffix + '.tmp')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        temporary.replace(target)
        manifest['files'][name] = {'url': url, 'bytes': target.stat().st_size,
                                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    common.atomic_write(args.output_dir / 'download_manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def prepare_data(args):
    grouped = defaultdict(list)
    validation = []
    counts = {}
    for split in ('test', 'validation'):
        rows = pq.read_table(args.input_dir / 'raw' / 'data' / f'{split}-00000-of-00001.parquet').to_pylist()
        counts[split] = dict(Counter(row['category'] for row in rows))
        seen = set()
        for row in rows:
            if row['question_id'] in seen:
                raise ValueError('duplicate official question_id')
            seen.add(row['question_id'])
            record = normalize(row, split)
            if not category_in_profile(row['category'], args.profile):
                continue
            if split == 'validation':
                validation.append(record)
            else:
                grouped[row['category']].append(record)
    partitions = {'train': [], 'dev': [], 'heldout': [], 'official_validation': validation}
    # Split duplicate question/option content as a unit to prevent exact-content leakage.
    for category, records in sorted(grouped.items()):
        units = defaultdict(list)
        for record in records:
            units[record['task']].append(record)
        ordered = sorted(units, key=lambda task: hashlib.sha256(f'{args.seed}\n{task}'.encode()).hexdigest())
        n = len(ordered)
        for i, task in enumerate(ordered):
            split = 'train' if i < int(n * .7) else 'dev' if i < int(n * .8) else 'heldout'
            partitions[split].extend(units[task])
    for split, records in partitions.items():
        for record in records:
            record['source_metadata'].update(local_split=split, profile=args.profile, split_seed=args.seed)
        common.atomic_write(args.output_dir / f'{split}.json', records)
    summary = {'profile': args.profile, 'official_category_counts': counts,
               'local_counts': {k: len(v) for k, v in partitions.items()},
               'local_category_counts': {k: dict(Counter(r['source_metadata']['category'] for r in v))
                                         for k, v in partitions.items()},
               'seed': args.seed, 'revision': REVISION,
               'note': 'Local train/dev/heldout partition official test; not official benchmark scores.'}
    common.atomic_write(args.output_dir / 'split_manifest.json', summary)
    print(json.dumps(summary, indent=2))


def source_rows(input_dir, split, limit=None):
    rows = common.load_array(input_dir / f'{split}.json')
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    return rows[:limit]


def main():
    # Reuse MATH selection, aggregation and validation CLI without global monkey-patching.
    if len(sys.argv) > 1 and sys.argv[1] in ('download', 'prepare-data'):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', choices=('download', 'prepare-data'))
        parser.add_argument('--input-dir', type=Path, default=Path('data/mmlu_pro'))
        parser.add_argument('--output-dir', type=Path, required=True)
        parser.add_argument(
            '--profile', choices=('knowledge', 'non_math', 'full'), default='knowledge'
        )
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args()
        (download if args.command == 'download' else prepare_data)(args)
        return
    args = common.parse_args()
    if getattr(args, 'rollouts', 5) <= 0:
        raise ValueError('rollouts must be positive')
    if args.command == 'prepare-anchors':
        common.prepare_anchors(args, row_loader=source_rows, pool_resolver=resolve_pool,
                               anchor_generator=role_anchors, evaluator='mmlu_pro')
    elif args.command == 'prepare-candidates':
        common.prepare_candidates(args, pool_resolver=resolve_pool, suite_generator=candidate_suite)
    else:
        {'select': common.select_questions, 'classify-anchors': common.classify_anchors,
         'aggregate': common.aggregate, 'validate': common.validate}[args.command](args)


if __name__ == '__main__':
    main()
