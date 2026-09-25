"""Check installation and bundled data; optionally check downloaded models and services."""
import argparse
import importlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def check_data():
    corpora = ['data/math/math_train_1000x12_avg5.json',
               'data/mmlu_pro/non_math/mmlu_pro_train_1000x12_avg5.json']
    for name in corpora:
        path = ROOT/name
        rows = json.loads(path.read_text())
        if len(rows) != 1000 or any(len(r['graphs']) != 12 for r in rows):
            raise ValueError(f'incomplete corpus: {name}')
        for reference in {r['node_pool'] for r in rows}:
            if not (path.parent/reference).is_file():
                raise FileNotFoundError(f'missing role pool: {reference}')
    benchmark = json.loads((ROOT/'data/math/benchmark400.json').read_text())['records']
    if len(benchmark) != 400 or len({r['task'] for r in benchmark}) != 400:
        raise ValueError('incomplete 400-question benchmark')
    print('Bundled corpora, role pools, and 400-question benchmark: OK')


def check_model(path):
    from transformers import AutoTokenizer, AutoConfig
    path = Path(path)
    AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.apply_chat_template([{'role':'user','content':'Check setup.'}],
                                  tokenize=True, add_generation_prompt=True, enable_thinking=False)
    index = path/'model.safetensors.index.json'
    weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not weights or any(not (path/name).is_file() or (path/name).stat().st_size == 0 for name in weights):
        raise FileNotFoundError(f'missing model weights in {path}; run python download_models.py')
    print(f'Model files: {path.name}: OK')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', action='store_true')
    p.add_argument('--services', action='store_true', help='Check both services after starting the SFT topology server')
    args = p.parse_args()
    for module in ('torch','transformers','peft','pyarrow','networkx','openai','httpx',
                   'math_verify','huggingface_hub','plasmas','train_sft','train_dpo'):
        importlib.import_module(module)
    print('Training dependencies and entry-point imports: OK')
    check_data()
    if args.models:
        check_model(os.environ.get('TOPOLOGY_MODEL', ROOT/'models/Qwen3-1.7B'))
        check_model(os.environ.get('AGENT_TOKENIZER', ROOT/'models/Qwen3-4B'))
    if args.services:
        for port, name in ((8000, 'qwen3-4b'), (8110, 'sft-refresh')):
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10) as response:
                models = json.load(response)
            if name not in {m['id'] for m in models['data']}:
                raise ValueError(f'{name} is not served on port {port}')
            print(f'Service {name}: OK')


if __name__ == '__main__':
    main()
