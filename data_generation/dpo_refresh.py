"""SFT candidate collection, five-seed MAS scoring, and accuracy-first DPO pairs."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plasmas.data import AGPJsonDataset, PairwiseRewardDataset
from plasmas.mas_runtime import VLLMChatBackend
from plasmas.qwen_topology_dpo import DPO_SYSTEM_PROMPT, topology_json, topology_messages
from evaluation.topology_io import extract_json, user_prompt, validate_topology


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def canonical(t):
    return json.dumps({'selected_agents': sorted(t['selected_agents']),
                       'edges': sorted(t['edges'])}, separators=(',', ':'))


def from_graph(g, nodes):
    ids = [str(n['id']) for n in nodes]
    return {'selected_agents': [n for n, m in zip(ids, g['mask']) if m == 0],
            'edges': [[ids[i], ids[j]] for i, row in enumerate(g['edge_weight'])
                      for j, v in enumerate(row) if v]}


def graph(t, nodes):
    ids = [str(n['id']) for n in nodes]
    edges = {tuple(e) for e in t['edges']}
    return {'mask': [int(n not in t['selected_agents']) for n in ids],
            'edge_weight': [[float((a, b) in edges) for b in ids] for a in ids]}


def perturbations(sources, existing, ids, seed):
    """Three distinct one-role toggles, each relative to its original source."""
    rng = random.Random(seed)
    seen = {canonical(t) for t in existing}
    result = []
    for base in sources:
        for node in rng.sample(ids[:-1], len(ids)-1):
            selected = set(base['selected_agents'])
            edges = set(map(tuple, base['edges']))
            if node in selected:
                selected.remove(node)
                edges = {(a,b) for a,b in edges if a != node and b != node}
            else:
                selected.add(node)
            edges |= {(a, ids[-1]) for a in selected if a != ids[-1]}
            t = {'selected_agents': sorted(selected), 'edges': [list(e) for e in sorted(edges)]}
            key = canonical(t)
            if key not in seen:
                seen.add(key); result.append(t)
                if len(result) == 3:
                    return result
    raise ValueError('cannot construct three distinct perturbations')


async def prepare(args):
    directory = args.output_dir
    fingerprint = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.data, args.roles, args.adapter/'adapter_model.safetensors',
                     args.adapter/'training_state.json')}
    family = getattr(args, 'task_family', 'math')
    test_path = getattr(args, 'heldout', None) or ROOT/'data/math/test_anchors_5_delta_levels.json'
    if family == 'mmlu_pro':
        if not getattr(args, 'heldout', None):
            raise ValueError('MMLU refresh requires an explicit local --heldout file')
        fingerprint['task_family'] = family
        fingerprint['heldout_sha256'] = hashlib.sha256(test_path.read_bytes()).hexdigest()
        fingerprint['prompt_sha256'] = hashlib.sha256(json.dumps(topology_messages('', [], family)).encode()).hexdigest()
    identity_path = directory/'input_fingerprints.json'
    if identity_path.exists() and read(identity_path) != fingerprint:
        raise ValueError('input or checkpoint changed; use a new output directory')
    write(identity_path, fingerprint)
    source = read(args.data)
    state = read(args.adapter / 'training_state.json')['args']
    if state.get('task_family', 'math') != family:
        raise ValueError('SFT and refresh task-family prompts must match')
    if Path(state['data']).resolve() != args.data.resolve():
        raise ValueError('SFT data path differs; cannot reproduce its validation split')
    pairs = PairwiseRewardDataset(AGPJsonDataset(args.data), **{
        k: state[k] for k in ('min_utility_gap', 'token_cost_weight', 'token_cost_scale',
                             'cost_tradeoff_reward_gap', 'reward_significance_z', 'fit_cost_weight')})
    groups = pairs.pair_groups
    if len(groups) != 1000:
        raise ValueError('paper protocol requires exactly 1000 problems')
    if state.get('validation_data') or state['validation_fraction'] != 0.1:
        raise ValueError('paper refresh requires the seeded 900/100 SFT split')
    order = list(range(len(groups)))
    random.Random(state['seed']).shuffle(order)
    heldout = set(order[:round(len(order) * state['validation_fraction'])])
    nodes = read(args.roles)['nodes']
    ids = [str(n['id']) for n in nodes]
    test_data = read(test_path)
    test_rows = test_data if isinstance(test_data, list) else test_data['records']
    test_tasks = {x['task'] for x in test_rows}
    backend = VLLMChatBackend('sft-refresh', tokenizer_path=args.tokenizer,
        base_url=args.topology_url, max_new_tokens=256, temperature=0.7, seed=7)
    greedy = VLLMChatBackend('sft-refresh', tokenizer_path=args.tokenizer,
        base_url=args.topology_url, max_new_tokens=256, temperature=0, seed=7)
    saved = directory / 'topologies.json'
    records = read(saved) if saved.exists() else []
    done = {r['group_index'] for r in records}
    for gi, group in enumerate(groups):
        if gi in done:
            continue
        original = source[int(group.fit_candidate.pair_group)]
        split_key = 'local_split' if family == 'mmlu_pro' else 'split'
        if original['source_metadata'][split_key] != 'train' or original['task'] in test_tasks:
            raise ValueError('train/test contamination in source')
        # Four base slots and three policy slots may describe the same DAG.
        # Keep slot identities; only perturbations must be new, distinct graphs.
        candidates = []
        def add(t, origin):
            t = validate_topology(t, nodes, require_difficulty=False)
            candidates.append({'topology': json.loads(canonical(t)), 'origins': [origin]})
        add(json.loads(topology_json(group.fit_candidate)), 'fit')
        add({'selected_agents': ['agent_5'], 'edges': []}, 'finalizer_only')
        expert = next(g for g in original['graphs'] if g['generator'] == 'expert_anchor')
        add(from_graph(expert, nodes), 'expert_anchor')
        add({'selected_agents': ids, 'edges': [[a, b] for i, a in enumerate(ids) for b in ids[i+1:]]}, 'full_forward')
        messages = topology_messages(group.task, nodes, family)
        raw = []
        for sample in range(3):
            response = await (greedy if sample == 0 else backend).generate(messages, seed=7 + gi * 3 + sample)
            item = {'text': response.text, 'seed': 7 + gi * 3 + sample,
                    'temperature': 0 if sample == 0 else 0.7}
            try:
                t = validate_topology(extract_json(response.text), nodes, require_difficulty=False)
            except (ValueError, TypeError, KeyError) as exc:
                write(directory/'invalid_generation.json', {'group_index': gi, **item, 'error': str(exc)})
                raise ValueError(f'group {gi}: invalid SFT proposal; inspect invalid_generation.json') from exc
            add(t, 'sft_greedy' if sample == 0 else 'sft_sample')
            raw.append({**item, 'valid': True})
        for t in perturbations([c['topology'] for i,c in enumerate(candidates) if i != 3],
                               [c['topology'] for c in candidates], ids, 42000 + gi):
            add(t, 'sft_mutation')
        if len(candidates) != 10:
            raise ValueError('expected ten valid candidate slots')
        records.append({'group_index': gi, 'split': 'validation' if gi in heldout else 'train',
            'task': group.task, 'reference_answer': original['reference_answer'],
            'source_index': original['source_index'], 'source_metadata': original['source_metadata'],
            'candidates': candidates, 'sft_generations': raw})
        write(saved, records)
        if len(records) % 10 == 0:
            print(f'topologies={len(records)}/{len(groups)}', flush=True)
    # Candidate graph scores are fresh: do not reuse historical anchor rewards.
    execution = []
    for r in records:
        gs = []
        for ci, c in enumerate(r['candidates']):
            for seed in range(42, 47):
                gs.append({**graph(c['topology'], nodes), 'id': f"q{r['group_index']}_c{ci}_s{seed}",
                    'generator': '+'.join(c['origins']), 'candidate_index': ci, 'sampling_seed': seed})
        execution.append({k: r[k] for k in ('task', 'reference_answer', 'source_index', 'source_metadata', 'group_index', 'split')})
        execution[-1].update(graphs=gs, node_pool=str(args.roles.resolve()), evaluator=family)
    write(directory / 'candidates.json', execution)
    manifest = {'questions': len(records), 'splits': dict(Counter(r['split'] for r in records)),
        'rollouts': sum(len(r['graphs']) for r in execution), 'mas_seeds': list(range(42,47)),
        'temperature': 0.7, 'max_new_tokens': 2048, 'model': 'qwen3-4b',
        'task_family': family,
        'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'adapter_sha256': hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
        'roles_sha256': hashlib.sha256(args.roles.read_bytes()).hexdigest(),
        'selection': f'existing {family} training questions; preserved SFT train/validation split',
        'preference_rule': 'accuracy gap >= 0.4, paired wins >= 3 and losses == 0; or both 5/5 and mean token saving >= 20%, saving on >= 4/5 seeds',
        'max_pairs_per_question': 3, 'quality_weight': 1, 'cost_weight': 0.25}
    write(directory/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def preference(a, b):
    wins = sum(x > y for x,y in zip(a['outcomes'], b['outcomes']))
    losses = sum(x < y for x,y in zip(a['outcomes'], b['outcomes']))
    gap = a['accuracy'] - b['accuracy']
    if gap >= 0.4 - 1e-9 and wins >= 3 and losses == 0:
        return 'quality', gap
    if a['accuracy'] == b['accuracy'] == 1:
        saving = 1 - a['mean_tokens'] / b['mean_tokens']
        if saving >= .2 - 1e-9 and sum(x < y for x,y in zip(a['tokens'], b['tokens'])) >= 4:
            return 'cost', saving
    return None


def finalize(args):
    directory = args.output_dir
    family = getattr(args, 'task_family', 'math')
    manifest_path = directory/'manifest.json'
    if manifest_path.exists() and read(manifest_path).get('task_family','math') != family:
        raise ValueError('finalize task family differs from candidate generation')
    export_dir = args.export_dir or directory
    if (args.allow_incomplete or args.exclude_unparsed) and export_dir.resolve() == directory.resolve():
        raise ValueError('partial export requires a separate --export-dir')
    export_dir.mkdir(parents=True, exist_ok=True)
    records = read(directory/'topologies.json')
    scored_bytes = (directory/'scored.json').read_bytes()
    scored_hash = hashlib.sha256(scored_bytes).hexdigest()
    scored = json.loads(scored_bytes)
    if len(records) != len(scored):
        raise ValueError('incomplete scored file')
    tokenizer = VLLMChatBackend('qwen3-4b', tokenizer_path=args.tokenizer).tokenizer
    nodes = read(args.roles)['nodes']
    output = {'train': [], 'validation': []}
    aggregated = []
    excluded = []
    for r, s in zip(records, scored):
        if r['group_index'] != s['group_index'] or r['task'] != s['task']:
            raise ValueError('scored/input identity mismatch')
        unfinished = [g['id'] for g in s['graphs'] if g.get('execution_status') != 'completed']
        unparsed = [g['id'] for g in s['graphs'] if g.get('execution_status') == 'completed' and g.get('prediction') is None]
        if args.exclude_unparsed and unfinished:
            raise ValueError('final export requires all rollouts completed')
        if (args.allow_incomplete and (unfinished or unparsed)) or (args.exclude_unparsed and unparsed):
            excluded.append({'group_index': r['group_index'], 'split': r['split'],
                             'unfinished': unfinished, 'unparsed': unparsed})
            continue
        stats = []
        for ci, c in enumerate(r['candidates']):
            runs = sorted([g for g in s['graphs'] if g['candidate_index'] == ci], key=lambda g:g['sampling_seed'])
            if [g['sampling_seed'] for g in runs] != list(range(42,47)):
                raise ValueError('missing or duplicated rollout seeds')
            if any(g.get('execution_status') != 'completed' or g.get('prediction') is None for g in runs):
                raise ValueError(f"incomplete or unparsed group={r['group_index']} candidate={ci}; no preferences exported")
            outcomes = [int(g['accuracy']) for g in runs]
            tokens = [g['total_input_tokens'] + g['total_output_tokens'] for g in runs]
            stats.append({**c, 'candidate_index': ci, 'outcomes': outcomes, 'tokens': tokens,
                'accuracy': sum(outcomes)/5, 'mean_tokens': sum(tokens)/5})
        possible = []
        for a,b in itertools.permutations(stats, 2):
            pref = preference(a,b)
            if pref:
                category, gap = pref
                on_policy = any(o.startswith('sft_') for o in a['origins'] + b['origins'])
                possible.append((category, gap, on_policy, a, b))
        possible.sort(key=lambda p:(p[0]=='quality', p[2], p[1]), reverse=True)
        # At most one cost pair; limit repeated question influence to three pairs.
        selected = []
        for p in possible:
            if p[0] == 'cost' and any(q[0]=='cost' for q in selected):
                continue
            selected.append(p)
            if len(selected) == 3:
                break
        messages = topology_messages(r['task'], nodes, family)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        for category,gap,on_policy,a,b in selected:
            output[r['split']].append({'prompt':prompt, 'messages':messages,
                'chosen':canonical(a['topology']), 'rejected':canonical(b['topology']),
                'weight':(1.0 if category=='quality' else .25)/len(selected),
                'category':category, 'ranking_gap':gap, 'group_index':r['group_index'],
                'source_index':r['source_index'], 'on_policy':on_policy,
                'chosen_evidence':a, 'rejected_evidence':b})
        aggregated.append({**{k:v for k,v in r.items() if k not in ('candidates','sft_generations')}, 'candidates':stats})
    write(export_dir/'aggregated.json', aggregated)
    for split, rows in output.items():
        path = export_dir/f'dpo_{split}.jsonl'
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        tmp.replace(path)
    report = {split:{'pairs':len(rows), 'questions':len({r['group_index'] for r in rows}),
        'categories':dict(Counter(r['category'] for r in rows)),
        'on_policy_pairs':sum(r['on_policy'] for r in rows)} for split,rows in output.items()}
    report['metadata'] = {'complete': not excluded, 'source_questions': len(records),
        'included_questions': len(aggregated), 'excluded_questions': len(excluded),
        'all_rollouts_completed': all(g.get('execution_status') == 'completed' for r in scored for g in r['graphs']),
        'exclusion_policy': 'exclude_whole_question_with_unparsed_answer' if args.exclude_unparsed else 'strict_or_partial',
        'source_scored_sha256': scored_hash}
    write(export_dir/'excluded.json', excluded)
    write(export_dir/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','finalize'])
    p.add_argument('--task-family',choices=['math','mmlu_pro'],default='math')
    p.add_argument('--heldout',type=Path,help='Explicit uncontaminated local evaluation set (required for MMLU)')
    p.add_argument('--data',type=Path,default=ROOT/'data/math/math_train_1000x12_avg5.json')
    p.add_argument('--roles',type=Path,default=ROOT/'data/node_pools/math_6_roles.json')
    p.add_argument('--adapter',type=Path,default=ROOT/'runs/math/sft/best')
    p.add_argument('--tokenizer',default='Qwen/Qwen3-1.7B')
    p.add_argument('--topology-url',default='http://127.0.0.1:8110/v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/math/refresh')
    p.add_argument('--export-dir',type=Path)
    p.add_argument('--allow-incomplete',action='store_true')
    p.add_argument('--exclude-unparsed',action='store_true',help='Require all rollouts completed, exclude whole questions with unparsed predictions without re-sampling.')
    args=p.parse_args()
    if args.stage=='prepare': asyncio.run(prepare(args))
    else: finalize(args)


if __name__=='__main__': main()
