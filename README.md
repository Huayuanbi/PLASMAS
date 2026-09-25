# PLASMAS: How Much Collaboration Do You Need in Multi-Agent System?

**PLASMAS** learns task-specific communication topologies for multi-agent systems. It combines supervised initialization with preference alignment to balance task performance and execution cost.

## Installation

Clone this repository into `PLASMAS/`, or download and extract its ZIP there. Use Linux, Python 3.11, and CUDA-capable GPUs for training and serving.

```bash
cd PLASMAS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Install the model server in a separate environment.
python3.11 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -r requirements-serving.txt

# Download Qwen3-1.7B and Qwen3-4B into models/.
python download_models.py
python check_setup.py --models
```

Run commands from the repository root. In each new training terminal, activate `.venv`. The examples allocate GPU 0 to training, GPU 1 to execution agents, and GPU 2 to topology serving; adjust these indices to your hardware.

Start the execution-agent service in a separate terminal and leave it running:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-vllm/bin/vllm serve "$PWD/models/Qwen3-4B" \
  --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768
```

## Quick Start

The example below runs MATH. Training data and the fixed 400-question benchmark are included.

### 1. Data Generation

Use the bundled training data:

```bash
export TRAIN_DATA="$PWD/data/math/math_train_1000x12_avg5.json"
```

Or generate a new corpus from raw MATH parquet files (`<category>/train-*.parquet`):

```bash
python generate_data.py train --task-family math \
  --input-dir /path/to/MATH --output-dir generated/math

export TRAIN_DATA="$PWD/generated/math/train_1000x12_avg5.json"
```

To rebuild the included 400-question benchmark:

```bash
python generate_data.py rebuild400 --output-dir generated/benchmark400
```

### 2. Phase 1: Supervised Initialization

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline.py sft \
  --task-family math --data "$TRAIN_DATA"
```

The selected adapter is saved to `runs/math/sft/best/`.

### 3. Phase 2: Preference Alignment

After Phase 1, serve its adapter in another terminal:

```bash
CUDA_VISIBLE_DEVICES=2 .venv-vllm/bin/vllm serve "$PWD/models/Qwen3-1.7B" \
  --served-model-name topology-base --host 127.0.0.1 --port 8110 \
  --enable-lora --max-lora-rank 16 \
  --lora-modules sft-refresh="$PWD/runs/math/sft/best" \
  --dtype bfloat16 --max-model-len 4096
```

Back in the training terminal, use the same `TRAIN_DATA` as Phase 1:

```bash
python check_setup.py --services
python pipeline.py refresh --task-family math --data "$TRAIN_DATA"
python pipeline.py score-refresh --task-family math
python pipeline.py pairs --task-family math
CUDA_VISIBLE_DEVICES=0 python pipeline.py dpo --task-family math
```

### 4. Test

Select a checkpoint on the validation set, then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline.py validation --task-family math

# MATH-5000
CUDA_VISIBLE_DEVICES=0 python pipeline.py test --task-family math \
  --source data/math/test_anchors_5_delta_levels.json

# 400-question topology sensitivity benchmark
CUDA_VISIBLE_DEVICES=0 python pipeline.py benchmark400 --task-family math
```

Results:

- `runs/math/validation/selection.json`
- `runs/math/test/selected/report.json`
- `runs/math/benchmark400/selected/sensitivity.json`

## MMLU-Pro

Use the bundled corpus:

```bash
export TRAIN_DATA="$PWD/data/mmlu_pro/non_math/mmlu_pro_train_1000x12_avg5.json"
```

Or generate a new one:

```bash
python generate_data.py prepare-mmlu --task-family mmlu_pro \
  --output-dir generated/mmlu_pro
python generate_data.py train --task-family mmlu_pro \
  --input-dir generated/mmlu_pro/non_math --output-dir generated/mmlu_train

export TRAIN_DATA="$PWD/generated/mmlu_train/train_1000x12_avg5.json"
```

Follow Phases 1 and 2 with `--task-family mmlu_pro`. Restart the topology service with `runs/mmlu_pro/sft/best` as its `sft-refresh` adapter. Then evaluate on the local heldout split:

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline.py validation --task-family mmlu_pro
CUDA_VISIBLE_DEVICES=0 python pipeline.py test --task-family mmlu_pro \
  --source data/mmlu_pro/non_math/heldout.json
```

Results are saved under `runs/mmlu_pro/`.
