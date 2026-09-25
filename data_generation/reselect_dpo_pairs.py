"""Reselect evidence-backed local cost/quality/structure preferences (no new rollouts)."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_generation.dpo_refresh import canonical, preference


def size(candidate):
    return len(candidate['topology']['selected_agents'])


def distance(a, b):
    x, y = a['topology'], b['topology']
    return (len(set(x['selected_agents']) ^ set(y['selected_agents'])) +
            len(set(map(tuple, x['edges'])) ^ set(map(tuple, y['edges']))))


def select_pairs(candidates):
    """Reserve independent slots; never manufacture a preference to fill a slot."""
    possible = []
    for a, b in itertools.permutations(candidates, 2):
        if canonical(a['topology']) == canonical(b['topology']):
            continue
        pref = preference(a, b)
        if pref:
            possible.append(dict(category=pref[0], ranking_gap=pref[1], a=a, b=b))
    selected, used = [], set()

    def pick(slot, allowed, rank):
        available = [p for p in possible if allowed(p) and
                     (canonical(p['a']['topology']), canonical(p['b']['topology'])) not in used]
        if not available:
            return
        p = min(available, key=lambda p: (*rank(p), canonical(p['a']['topology']), canonical(p['b']['topology'])))
        used.add((canonical(p['a']['topology']), canonical(p['b']['topology'])))
        selected.append({**p, 'selection_slot': slot})

    # Do not allow the largest cost saving (usually 1-vs-6) to crowd out 1-vs-2.
    pick('local_cost', lambda p: p['category'] == 'cost' and size(p['a']) < size(p['b']),
         lambda p: (size(p['b'])-size(p['a']), size(p['a']), distance(p['a'],p['b']), -p['ranking_gap']))
    # Quality can prefer fewer nodes too. Among reliable winners, keep the smallest
    # winner and prefer a nearby loser; do not always choose the largest accuracy gap.
    pick('quality', lambda p: p['category'] == 'quality',
         lambda p: (-p['a']['accuracy'], size(p['a']), abs(size(p['a'])-size(p['b'])),
                    distance(p['a'],p['b']), p['a']['mean_tokens'], -p['ranking_gap']))
    pick('same_size_structure', lambda p: size(p['a']) == size(p['b']),
         lambda p: (p['category'] != 'quality', size(p['a']), distance(p['a'],p['b']), -p['ranking_gap']))
    return selected


def summarize(rows):
    def stats(rs):
        weights = sum(r['weight'] for r in rs)
        return {'pairs': len(rs), 'questions': len({r['group_index'] for r in rs}),
                'categories': dict(Counter(r['category'] for r in rs)),
                'slots': dict(Counter(r.get('selection_slot','legacy') for r in rs)),
                'category_weights': {c: sum(r['weight'] for r in rs if r['category']==c) for c in ['quality','cost']},
                'chosen_rejected_nodes': dict(sorted(Counter(
                    f"{len(json.loads(r['chosen'])['selected_agents'])}>{len(json.loads(r['rejected'])['selected_agents'])}" for r in rs).items())),
                'weighted_chosen_nodes': sum(r['weight']*len(json.loads(r['chosen'])['selected_agents']) for r in rs)/weights if weights else None,
                'weighted_rejected_nodes': sum(r['weight']*len(json.loads(r['rejected'])['selected_agents']) for r in rs)/weights if weights else None}
    return {'overall': stats(rows), 'by_level': {level: stats([r for r in rows if r['math_level']==level])
             for level in sorted({r['math_level'] for r in rows})}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/math/refresh/filtered')
    p.add_argument('--output', type=Path, default=ROOT/'runs/math/pairs')
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--task-family', choices=['math','mmlu_pro'], default='math')
    p.add_argument('--roles',type=Path,default=ROOT/'data/node_pools/math_6_roles.json')
    args = p.parse_args()
    from transformers import AutoTokenizer
    from plasmas.qwen_topology_dpo import topology_messages
    from evaluation.topology_io import user_prompt, validate_topology
    nodes = json.loads(args.roles.read_text())['nodes']
    source_path = args.source/'aggregated.json'
    records = json.loads(source_path.read_text())
    if args.output.resolve()==args.source.resolve():
        raise ValueError('v2 must not overwrite v1')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    output = {'train': [], 'validation': []}
    seen = set()
    for record in records:
        split_key='local_split' if args.task_family=='mmlu_pro' else 'split'
        if record['task'] in seen or record['source_metadata'][split_key] != 'train':
            raise ValueError('duplicate question or non-training source')
        seen.add(record['task'])
        for c in record['candidates']:
            validate_topology(c['topology'], nodes, require_difficulty=False)
            if len(c['outcomes']) != 5 or len(c['tokens']) != 5 or any(x not in [0,1] for x in c['outcomes']):
                raise ValueError('invalid five-seed evidence')
            if abs(c['accuracy']-sum(c['outcomes'])/5)>1e-9 or abs(c['mean_tokens']-sum(c['tokens'])/5)>1e-6:
                raise ValueError('aggregate disagrees with evidence')
        selected = select_pairs(record['candidates'])
        messages = topology_messages(record['task'],nodes,args.task_family)
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        for pref in selected:
            a,b = pref['a'],pref['b']
            output[record['split']].append({
                'messages':messages,'prompt':prompt,'chosen':canonical(a['topology']),'rejected':canonical(b['topology']),
                'weight':(1. if pref['category']=='quality' else .25)/len(selected),
                'category':pref['category'],'ranking_gap':pref['ranking_gap'],'selection_slot':pref['selection_slot'],
                'group_index':record['group_index'],'source_index':record['source_index'],
                'math_level':record['source_metadata'].get('level','not_applicable'),
                **({'subject':record['source_metadata']['category']} if args.task_family=='mmlu_pro' else {}),
                'on_policy':any(o.startswith('sft_') for o in a['origins']+b['origins']),
                'chosen_evidence':a,'rejected_evidence':b})
    report = {'version':'local_pairs_v2','source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'rules':{'slots':['local_cost','quality','same_size_structure'], 'max_pairs_per_question':3,
                       'quality_weight':1.,'cost_weight':.25,'normalization':'divide by selected pairs per question',
                       'new_rollouts':0,'label_rule':'unchanged from v1; five-seed heuristic, not proof of equivalence'},
              **{split:summarize(rs) for split,rs in output.items()}}
    if args.task_family=='mmlu_pro':
        report['task_family']='mmlu_pro'
        report['roles_sha256']=hashlib.sha256(args.roles.read_bytes()).hexdigest()
        for split,rows in output.items():
            report[split]['by_subject']={s:summarize([r for r in rows if r['subject']==s])['overall']
                                         for s in sorted({r['subject'] for r in rows})}
    # Identical reruns are harmless; changed outputs require a new directory.
    payloads = {'report.json':json.dumps(report,ensure_ascii=False,indent=2)+'\n'}
    payloads.update({f'dpo_{split}.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs) for split,rs in output.items()})
    for name,content in payloads.items():
        path=args.output/name
        if path.exists() and path.read_text()!=content:
            raise ValueError(f'refusing to overwrite different v2 data: {path}')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,content in payloads.items():
        path=args.output/name
        if not path.exists():
            tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(content);tmp.replace(path)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
