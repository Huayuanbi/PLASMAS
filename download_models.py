"""Download the public Qwen3 models to this repository's models directory."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {'topology': 'Qwen/Qwen3-1.7B', 'agents': 'Qwen/Qwen3-4B'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['all', *MODELS], default='all')
    p.add_argument('--output-dir', type=Path, default=ROOT/'models')
    p.add_argument('--tokenizer-only', action='store_true', help='Download tokenizer/config files for CPU data checks only')
    args = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    for name, repository in MODELS.items():
        if args.model not in ('all', name):
            continue
        revision = HfApi().model_info(repository, token=False).sha
        destination = args.output_dir/repository.split('/')[-1]
        patterns = ['*.json', '*.txt', '*.model', '*.tiktoken']
        if not args.tokenizer_only:
            patterns.append('*.safetensors')
        snapshot_download(repository, revision=revision, local_dir=destination,
                          allow_patterns=patterns, token=False)
        (destination/'download_manifest.json').write_text(json.dumps({
            'repository': repository, 'revision': revision,
            'tokenizer_only': args.tokenizer_only,
        }, indent=2)+'\n')
        print(f'{name}: {destination}', flush=True)


if __name__ == '__main__':
    main()
