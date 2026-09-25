"""Rebuild initial 1000x12x5 corpora or the fixed MATH-400 benchmark."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands(args):
    family = args.task_family
    work = args.output_dir.resolve()
    roles = ROOT/f'data/node_pools/{family}_6_roles.json'
    script = f'data_generation/{family}_pipeline.py'
    def py(path, *options):
        return [sys.executable, '-u', str(ROOT/path), *map(str,options)]
    def stage(name, *options):
        return py(script, name, *options)
    def score(name):
        return py('data_generation/run_mas.py', '--input', work/f'{name}.json',
                  '--output', work/f'{name}_scored.json', '--backend', 'vllm',
                  '--model', args.agent_model, '--tokenizer', args.agent_tokenizer,
                  '--base-url', args.agent_url, '--temperature', .7, '--max-new-tokens', 2048,
                  '--seed', 42, '--evaluator', family, '--concurrency', args.concurrency,
                  '--resume', '--checkpoint-every', 10)
    if args.stage == 'rebuild400':
        yield py('evaluation/build_benchmark400.py', '--input', args.anchor_summary,
                 '--output', work/'benchmark400.json', '--seed', 20260901)
        return
    if args.stage == 'prepare-mmlu':
        if family != 'mmlu_pro':
            raise ValueError('prepare-mmlu requires --task-family mmlu_pro')
        yield stage('download', '--output-dir', work/'raw_source')
        yield stage('prepare-data', '--input-dir', work/'raw_source', '--output-dir', work/'non_math',
                    '--profile', 'non_math', '--seed', 42)
        return
    if args.input_dir is None:
        raise ValueError('raw generation requires --input-dir')
    if args.stage == 'test-anchors' and family != 'math':
        raise ValueError('the 400-question benchmark uses MATH test anchors')
    split = 'test' if args.stage == 'test-anchors' else 'train'
    yield stage('prepare-anchors', '--input-dir', args.input_dir.resolve(), '--split', split,
                '--node-pool', roles, '--output', work/'anchors.json', '--rollouts', 5, '--seed', 42)
    yield score('anchors')
    if args.stage == 'test-anchors':
        yield stage('classify-anchors', '--input', work/'anchors_scored.json',
                    '--output', work/'anchor_summary.json', '--rollouts', 5)
        yield py('evaluation/build_benchmark400.py', '--input', work/'anchor_summary.json',
                 '--output', work/'benchmark400.json', '--seed', 20260901)
        return
    # The MMLU historical screen uses delta >= .4; common CLI uses strict >.
    threshold = '.5' if family == 'math' else '.399999999999'
    yield stage('select', '--anchor-scored', work/'anchors_scored.json', '--output', work/'selected.json',
                '--total-questions', 1000, '--delta-ratio', .5, '--high-ratio', .35, '--low-ratio', .15,
                '--delta-threshold', threshold, '--high-threshold', .8, '--low-threshold', .4, '--seed', 42)
    yield stage('prepare-candidates', '--selection', work/'selected.json',
                '--anchor-scored', work/'anchors_scored.json', '--node-pool', roles,
                '--output', work/'candidates.json', '--rollouts', 5, '--extra-candidates', 10,
                '--random-count', 5, '--seed', 42)
    yield score('candidates')
    yield stage('aggregate', '--input', work/'candidates_scored.json', '--output', work/'train_1000x12_avg5.json',
                '--topologies', 12, '--rollouts', 5)
    yield stage('validate', '--input', work/'train_1000x12_avg5.json', '--questions', 1000,
                '--topologies', 12, '--rollouts', 5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train','prepare-mmlu','test-anchors','rebuild400'])
    p.add_argument('--task-family', choices=['math','mmlu_pro'], default='math')
    p.add_argument('--input-dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--anchor-summary', type=Path, default=ROOT/'data/math/test_anchors_5_delta_levels.json')
    p.add_argument('--agent-tokenizer', default=os.environ.get('AGENT_TOKENIZER',str(ROOT/'models/Qwen3-4B')))
    p.add_argument('--agent-model', default='qwen3-4b')
    p.add_argument('--agent-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for command in commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
