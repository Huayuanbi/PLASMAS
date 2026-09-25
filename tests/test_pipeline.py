import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.dpo_refresh import canonical, perturbations, preference
from data_generation.reselect_dpo_pairs import select_pairs
from evaluation.evaluate import choose, load_records
from evaluation.topology_io import validate_topology
from evaluation.build_benchmark400 import difficulty_group, CATEGORY_QUOTAS
from pipeline import commands


class PaperProtocolTests(unittest.TestCase):
    def test_raw_mmlu_heldout_indices(self):
        rows = load_records(ROOT/'data/mmlu_pro/non_math/heldout.json', 'test')
        self.assertEqual(len({r['source_index'] for r in rows}), len(rows))
        self.assertTrue(all(r['source_metadata']['local_split']=='heldout' for r in rows))

    def test_benchmark_identity_and_quotas(self):
        from collections import Counter
        rows = json.loads((ROOT/'data/math/benchmark400.json').read_text())['records']
        self.assertEqual(len({r['source_index'] for r in rows}), 400)
        for row in rows:
            self.assertEqual(difficulty_group(row), row['difficulty_group'])
            self.assertEqual(row['source_metadata']['split'], 'test')
        for group in {r['difficulty_group'] for r in rows}:
            self.assertEqual(Counter(r['source_metadata']['type'] for r in rows if r['difficulty_group']==group), CATEGORY_QUOTAS)

    def test_perturbations_are_distinct_and_legal(self):
        nodes = json.loads((ROOT/'data/node_pools/math_6_roles.json').read_text())['nodes']
        ids = [n['id'] for n in nodes]
        source = {'selected_agents':['agent_5'], 'edges':[]}
        result = perturbations([source], [source], ids, 7)
        self.assertEqual(len(set(map(canonical,result))), 3)
        self.assertEqual(result, perturbations([source], [source], ids, 7))
        for t in result:
            validate_topology(t, nodes, require_difficulty=False)
            self.assertEqual(len(t['selected_agents']),2)
        self.assertEqual(source, {'selected_agents':['agent_5'], 'edges':[]})

    def test_conservative_preferences(self):
        def candidate(outcomes, tokens):
            return dict(outcomes=outcomes, tokens=tokens,
                        accuracy=sum(outcomes)/5, mean_tokens=sum(tokens)/5)
        a=candidate([1]*5,[80]*5); b=candidate([1]*5,[100]*5)
        self.assertEqual(preference(a,b)[0], 'cost')
        self.assertIsNone(preference(candidate([1]*5,[81]*5),b))
        a=candidate([1,1,1,0,0],[80]*5); b=candidate([0,0,0,0,0],[100]*5)
        self.assertEqual(preference(a,b)[0], 'quality')
        self.assertIsNone(preference(a,candidate([0,0,0,1,0],[100]*5)))

    def test_local_pair_slots_do_not_duplicate(self):
        candidates=[]
        for size,outcomes,tokens in [(1,[1]*5,50),(2,[1]*5,100),(3,[0]*5,200)]:
            ids=[f'agent_{i}' for i in range(size-1)]+['agent_5']
            candidates.append(dict(topology={'selected_agents':ids,'edges':[[n,'agent_5'] for n in ids[:-1]]},
                outcomes=outcomes,tokens=[tokens]*5,accuracy=sum(outcomes)/5,mean_tokens=tokens))
        selected=select_pairs(candidates)
        self.assertEqual([p['selection_slot'] for p in selected],['local_cost','quality'])
        self.assertEqual(len(selected[0]['a']['topology']['selected_agents']),1)
        self.assertEqual(len(selected[0]['b']['topology']['selected_agents']),2)

    def test_selection_and_test_leakage_guard(self):
        common=dict(split='validation',questions=100,source_sha256='source',roles_sha256='roles',
                    model='base',sft_sha256='sft',task_family='math',mas_seeds=[142,143,144,145,146],
                    mas_temperature=.7,mas_max_tokens=2048,topology_temperature=0,topology_max_tokens=256,
                    topology_seed=7,corrective_attempts=1,invalid_topologies=0)
        rows=[dict(common,name='sft',epoch=0,checkpoint=None,accuracy=.8,mean_total_tokens=200),
              dict(common,name='epoch2',epoch=2,checkpoint='epoch2',accuracy=.79,mean_total_tokens=100),
              dict(common,name='epoch3',epoch=3,checkpoint='epoch3',accuracy=.81,mean_total_tokens=180)]
        self.assertEqual(choose(rows)['selected'],'epoch3')
        rows[1]['accuracy']=.80
        self.assertEqual(choose(rows)['selected'],'epoch2')
        rows[1]['invalid_topologies']=1
        self.assertEqual(choose(rows)['selected'],'epoch3')
        rows[0]['accuracy']=.99
        self.assertFalse(choose(rows)['accuracy_constraint_met'])
        rows[0]['split']='test'
        with self.assertRaises(ValueError):choose(rows)

    def test_paper_training_commands(self):
        args=SimpleNamespace(task_family='math',work_dir=Path('/tmp/paper_protocol'),data=None,
                             topology_model='local/Qwen3-1.7B',stage='sft')
        sft=list(commands(args))[0]
        self.assertEqual(sft[sft.index('--lr')+1],'5e-5')
        self.assertNotIn('--fit-cost-weight',sft)
        args.stage='dpo'
        dpo=list(commands(args))[0]
        self.assertEqual(dpo[dpo.index('--graph-weight')+1],'0')
        self.assertEqual(dpo[dpo.index('--sft-weight')+1],'0.1')
        self.assertEqual(dpo[dpo.index('--lr')+1],'1e-6')

    def test_tied_candidates_still_receive_sft_demonstration(self):
        from plasmas.data import AGPJsonDataset, PairwiseRewardDataset
        # Use a real schema, then make every candidate equally accurate/costly.
        dataset=AGPJsonDataset(ROOT/'data/math/math_train_1000x12_avg5.json')
        dataset.examples=dataset.examples[:12]
        from dataclasses import replace
        dataset.examples=[replace(e,reward=1.) for e in dataset.examples]
        pairs=PairwiseRewardDataset(dataset, token_cost_weight=0.)
        self.assertEqual(len(pairs.pair_groups),1)
        self.assertEqual(len(pairs.pair_groups[0].pairs),0)
        self.assertEqual(pairs.pair_groups[0].fit_candidate.total_token_cost,
                         min(e.total_token_cost for e in dataset.examples))

    def test_dpo_gradients_and_completion_mask(self):
        import torch
        from training_utils import dpo_losses, completion_logps
        chosen=torch.tensor([-.5],requires_grad=True)
        rejected=torch.tensor([-1.],requires_grad=True)
        loss,_=dpo_losses(chosen,rejected,torch.tensor([-.5]),torch.tensor([-1.]),.1)
        loss.sum().backward()
        self.assertLess(chosen.grad.item(),0.)
        self.assertGreater(rejected.grad.item(),0.)
        logits=torch.zeros(1,3,4,requires_grad=True)
        labels=torch.tensor([[-100,-100,2]])
        value=completion_logps(logits,labels)
        self.assertAlmostEqual(value.item(),-1.38629436,places=6)

    def test_sensitivity_constant_graphs(self):
        from evaluation.sensitivity import summarize
        rows=[dict(task=f'{group}-{i}',difficulty_group=group,
                   topology={'selected_agents':['agent_5'],'edges':[]})
              for group in ('easy','medium','collaboration_required','hard_unsolved') for i in range(100)]
        result=summarize(rows)
        self.assertEqual(len(result['comparisons']),6)
        for value in result['comparisons'].values():
            self.assertEqual(value['gap'],0.)
            self.assertEqual(value['bootstrap_95_ci'],[0.,0.])
            self.assertEqual(value['holm_p'],1.)
            self.assertIsNone(value['cohen_d'])


if __name__ == '__main__':
    unittest.main()
