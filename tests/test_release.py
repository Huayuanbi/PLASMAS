"""CPU-only checks for a fresh checkout; no pretrained weights or services required."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class ReleaseTests(unittest.TestCase):
    def test_model_check_rejects_tokenizer_only_download(self):
        from unittest.mock import patch
        from check_setup import check_model
        with tempfile.TemporaryDirectory() as tmp, \
             patch('transformers.AutoConfig.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'):
            with self.assertRaisesRegex(FileNotFoundError, 'download_models.py'):
                check_model(tmp)

    def test_actual_sft_save_merge_and_dpo_backward_on_cpu(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from train_sft import SFTCollator, SFTExample, completion_loss
        from train_dpo import batch_terms
        from training_utils import DPOCollator
        from plasmas.qwen_topology_dpo import TopologyDPOExample

        torch.manual_seed(7)
        raw = Tokenizer(models.WordLevel({'[PAD]':0, '[UNK]':1, '[EOS]':2,
                                         'prompt':3, 'chosen':4, 'rejected':5}, unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
                                            unk_token='[UNK]', eos_token='[EOS]')
        config = Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, max_position_embeddings=64, pad_token_id=0, eos_token_id=2)
        base = Qwen3ForCausalLM(config)
        lora = LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, lora_dropout=0.,
                          target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        sft = get_peft_model(base, lora)
        batch = SFTCollator(tokenizer, 32)([(0, SFTExample('prompt', 'chosen', 0))])
        loss, _ = completion_loss(sft, batch, 0)
        loss.backward()
        optimizer = torch.optim.AdamW((p for p in sft.parameters() if p.requires_grad), lr=1e-3)
        optimizer.step(); optimizer.zero_grad()
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as tmp:
            sft.save_pretrained(tmp)
            # Reload the saved adapter onto the same original backbone weights.
            original = sft.unload()
            merged = PeftModel.from_pretrained(original, tmp).merge_and_unload()
            policy = get_peft_model(merged, lora)
            example = TopologyDPOExample('prompt', 'chosen', 'rejected', 1., 'quality', .6, 0)
            batch = DPOCollator(tokenizer, 32)([(0, example)])
            args = SimpleNamespace(activation_offload_threshold=0, beta=.1, sft_weight=.1, graph_weight=0.)
            terms = batch_terms(policy, None, batch, [], args)
            terms['total'].sum().backward()
            self.assertTrue(torch.isfinite(terms['total']).all())
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum())
                                for n,p in policy.named_parameters() if 'lora_' in n))
            self.assertFalse(any(p.requires_grad for n,p in policy.named_parameters() if 'lora_' not in n))

    def test_evaluation_report_counts_invalid_topology_and_total_cost(self):
        from evaluation.evaluate import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = dict(mas_seeds=[42], name='test', split='test')
            graph = dict(sampling_seed=42, mask=[1,1,1,1,1,0], edge_weight=[[0]*6 for _ in range(6)])
            valid = dict(source_index=0, task='valid task', reference_answer='1',
                         generation=dict(input_tokens=10, output_tokens=5),
                         topology={'selected_agents':['agent_5'],'edges':[]}, graphs=[graph])
            invalid = dict(source_index=1, task='invalid task', reference_answer='2',
                           generation=dict(input_tokens=12, output_tokens=8), topology=None, graphs=[])
            scored = json.loads(json.dumps([valid, invalid]))
            scored[0]['graphs'][0].update(execution_status='completed', accuracy=1, prediction='1',
                                         total_input_tokens=60, total_output_tokens=40)
            for name, value in [('identity',identity),('candidates',[valid,invalid]),('scored',scored)]:
                (root/f'{name}.json').write_text(json.dumps(value))
            summarize(SimpleNamespace(output=root))
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report['accuracy'], .5)
            self.assertEqual(report['invalid_topologies'], 1)
            self.assertEqual(report['mean_total_tokens'], 67.5)


if __name__ == '__main__':
    unittest.main()
