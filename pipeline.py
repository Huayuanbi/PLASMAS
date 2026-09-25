"""Paper-configured PLASMAS stages. Run from this directory; --dry-run prints commands."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands(args):
    family = args.task_family
    work = args.work_dir.resolve()
    roles = ROOT/f'data/node_pools/{family}_6_roles.json'
    data = args.data or ROOT/('data/math/math_train_1000x12_avg5.json' if family == 'math' else
                 'data/mmlu_pro/non_math/mmlu_pro_train_1000x12_avg5.json')
    heldout = ROOT/('data/math/test_anchors_5_delta_levels.json' if family == 'math' else
                    'data/mmlu_pro/non_math/heldout.json')
    sft = work/'sft/best'
    refresh = work/'refresh'
    def py(script, *options):
        return [sys.executable, '-u', str(ROOT/script), *map(str, options)]
    def score(source, output):
        return py('data_generation/run_mas.py', '--input', source, '--output', output,
                  '--backend', 'vllm', '--base-url', args.agent_url, '--model', args.agent_model,
                  '--tokenizer', args.agent_tokenizer, '--temperature', .7, '--max-new-tokens', 2048,
                  '--seed', 42, '--evaluator', family, '--concurrency', args.concurrency,
                  '--checkpoint-every', 10, '--resume')
    if args.stage == 'sft':
        yield py('train_sft.py', '--data', data, '--task-family', family, '--model', args.topology_model,
                 '--output-dir', work/'sft', '--lr', '5e-5', '--epochs', 10,
                 '--batch-size', 1, '--gradient-accumulation-steps', 16,
                 '--lora-r', 16, '--lora-alpha', 32, '--seed', 7, '--validation-fraction', .1)
    elif args.stage == 'refresh':
        yield py('data_generation/dpo_refresh.py', 'prepare', '--task-family', family,
                 '--data', data, '--roles', roles, '--adapter', sft, '--heldout', heldout,
                 '--tokenizer', args.topology_model, '--topology-url', args.topology_url,
                 '--output-dir', refresh)
    elif args.stage == 'score-refresh':
        yield score(refresh/'candidates.json', refresh/'scored.json')
    elif args.stage == 'pairs':
        yield py('data_generation/dpo_refresh.py', 'finalize', '--task-family', family,
                 '--roles', roles, '--tokenizer', args.topology_model, '--output-dir', refresh,
                 '--export-dir', refresh/'filtered', '--exclude-unparsed')
        yield py('data_generation/reselect_dpo_pairs.py', '--task-family', family,
                 '--source', refresh/'filtered', '--output', work/'pairs',
                 '--model', args.topology_model, '--roles', roles)
    elif args.stage == 'dpo':
        yield py('train_dpo.py', '--model', args.topology_model, '--sft-adapter', sft,
                 '--train-pairs', work/'pairs/dpo_train.jsonl',
                 '--validation-pairs', work/'pairs/dpo_validation.jsonl',
                 '--output-dir', work/'dpo', '--epochs', 10, '--lr', '1e-6', '--beta', .1,
                 '--sft-weight', .1, '--graph-weight', 0, '--batch-size', 1,
                 '--gradient-accumulation-steps', 16, '--seed', 7, '--resume-latest')
    elif args.stage in ('validation', 'test', 'benchmark400'):
        validation = args.stage == 'validation'
        if args.stage == 'benchmark400' and family != 'math':
            raise ValueError('the 400-question collaboration benchmark uses the MATH role pool')
        source = refresh/'topologies.json' if validation else (
            ROOT/'data/math/benchmark400.json' if args.stage == 'benchmark400' else args.source)
        if source is None:
            raise ValueError('test requires --source (list of task/reference/source_metadata records)')
        policies = [('sft', 0, None)] + [(f'epoch{e}',e,work/f'dpo/epoch{e}') for e in (2,3,5,8,10)] if validation else [
            ('selected', 0, Path(json.loads((work/'validation/selection.json').read_text())['checkpoint']))]
        reports = []
        for name, epoch, checkpoint in policies:
            output = work/args.stage/name
            command = py('evaluation/evaluate.py', 'generate', '--source', source, '--roles', roles,
                        '--model', args.topology_model, '--sft-adapter', sft, '--task-family', family,
                        '--split', 'validation' if validation else 'test', '--name', name,
                        '--epoch', epoch, '--output', output)
            if checkpoint is not None:
                command += ['--checkpoint', str(checkpoint)]
            yield command
            if args.stage != 'benchmark400':
                yield score(output/'candidates.json', output/'scored.json')
                yield py('evaluation/evaluate.py', 'summarize', '--output', output)
                reports.append(output/'report.json')
            else:
                yield py('evaluation/sensitivity.py', '--input', output/'topologies.json',
                         '--output', output/'sensitivity.json')
        if validation:
            yield py('evaluation/evaluate.py', 'select', '--reports', *reports, '--output', work/'validation')
    else:
        raise ValueError(args.stage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['sft','refresh','score-refresh','pairs','dpo','validation','test','benchmark400'])
    p.add_argument('--task-family', choices=['math','mmlu_pro'], default='math')
    p.add_argument('--work-dir', type=Path)
    p.add_argument('--data', type=Path, help='Optional regenerated 1000x12 avg5 corpus')
    p.add_argument('--topology-model', default=os.environ.get('TOPOLOGY_MODEL',str(ROOT/'models/Qwen3-1.7B')))
    p.add_argument('--agent-tokenizer', default=os.environ.get('AGENT_TOKENIZER',str(ROOT/'models/Qwen3-4B')))
    p.add_argument('--agent-model', default='qwen3-4b')
    p.add_argument('--agent-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--topology-url', default='http://127.0.0.1:8110/v1')
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--source', type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    args.work_dir = args.work_dir or ROOT/'runs'/args.task_family
    if args.data is not None:
        args.data = args.data.resolve()
    for command in commands(args):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
