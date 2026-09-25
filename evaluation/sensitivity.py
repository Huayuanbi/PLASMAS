"""Appendix A: 400-question node/edge gaps, bootstrap, permutation and Holm tests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def comparison(target, easy):
    a, b = np.asarray(target, dtype=float), np.asarray(easy, dtype=float)
    gap = float(a.mean()-b.mean())
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    rng = np.random.default_rng(7)
    boot = [rng.choice(a, len(a), replace=True).mean()-rng.choice(b, len(b), replace=True).mean()
            for _ in range(5000)]
    rng = np.random.default_rng(7)
    joined = np.concatenate([a,b])
    extreme = 0
    for _ in range(5000):
        shuffled = rng.permutation(joined)
        delta = shuffled[:len(a)].mean()-shuffled[len(a):].mean()
        extreme += abs(delta) >= abs(gap)-1e-12
    return dict(gap=gap, cohen_d=float(gap/pooled) if pooled > 0 else None,
                bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist(),
                permutation_p=(1+extreme)/5001)


def summarize(rows):
    groups = ('easy', 'medium', 'collaboration_required', 'hard_unsolved')
    if Counter(r['difficulty_group'] for r in rows) != Counter({g:100 for g in groups}):
        raise ValueError('expected the fixed four groups of 100 questions')
    if len({r['task'] for r in rows}) != 400:
        raise ValueError('duplicate questions')
    if any(r['topology'] is None for r in rows):
        raise ValueError('invalid topology: cannot substitute a graph in sensitivity statistics')
    counts = {g: {'nodes': [len(r['topology']['selected_agents']) for r in rows if r['difficulty_group']==g],
                  'edges': [len(r['topology']['edges']) for r in rows if r['difficulty_group']==g]}
              for g in groups}
    comparisons = {f'{g}_vs_easy/{metric}': comparison(counts[g][metric], counts['easy'][metric])
                   for g in groups[1:] for metric in ('nodes','edges')}
    previous = 0.
    for rank, name in enumerate(sorted(comparisons, key=lambda k: comparisons[k]['permutation_p'])):
        previous = min(1., max(previous, (6-rank)*comparisons[name]['permutation_p']))
        comparisons[name]['holm_p'] = previous
    return dict(questions=400, bootstrap_resamples=5000, permutations=5000, seed=7,
                group_means={g:{k:float(np.mean(v)) for k,v in stats.items()} for g,stats in counts.items()},
                comparisons=comparisons)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = summarize(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
