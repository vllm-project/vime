# MemAgent on vime

[MemAgent](https://arxiv.org/abs/2507.02259) (*Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent*) is an **RL-based memory agent** workflow for very long documents. It splits a document into chunks, reads them sequentially, and compresses key information into a **fixed-size memory** (overwrite policy). After all chunks are processed, the model answers using only the problem statement and the memory, with the final answer in `\boxed{}`. Because memory size stays constant, inference scales **linearly** \(O(N)\) with document length—without changing model architecture or positional encodings.

The paper trains end-to-end with **Multi-Conv DAPO** (this example uses **GRPO**) on multi-turn, context-independent trajectories with verifiable rewards on HotpotQA and RULER. Qwen2.5-7B trained on 32K documents generalizes to million-token QA with near-lossless performance on RULER-HQA.

This example reproduces that pipeline on vime: HotpotQA multi-turn rollout, GRPO training, and RULER-HQA evaluation, with vLLM as the inference backend.

## Files

| File | Description |
|------|-------------|
| `rollout.py` | Multi-turn MemAgent rollout + HotpotQA reward |
| `rollout_client.py` | vLLM router client for multi-turn turns |
| `custom_convert.py` | Unroll trajectories for GRPO training |
| `prepare_data.py` | HotpotQA parquet/HF → JSONL |
| `eval_ruler_hqa.py` | RULER-HQA evaluation script |

## Launch scripts

Run from **vime repo root** (`cd vime`):

| Script | Purpose |
|--------|---------|
| `run-qwen3-4b-train.sh` | GRPO training (default 100 steps) |
| `run-eval.sh` | RULER-HQA eval (vLLM serve + `eval_ruler_hqa.py`) |
| `convert-to-hf.sh` | Megatron `iter_*` → HuggingFace |
| `prepare-eval-data.sh` | Check/download `eval_{length}.json` |

Shared setup lives in `_common.sh` (paths, MemAgent env vars, Ray launch).

### Quick start

#### Data download

Training and evaluation data come from the MemAgent HuggingFace dataset **[BytedTsinghua-SIA/hotpotqa](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa)**.

**Training set** — download the `train` split and convert to vime JSONL:

```bash
pip install datasets huggingface_hub pandas pyarrow

# Optional: use a mirror if huggingface.co is slow
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /data/datasets/hotpotqa_slime

python examples/mem_agent/prepare_data.py \
  --hf-dataset BytedTsinghua-SIA/hotpotqa \
  --hf-split train \
  --output /data/datasets/hotpotqa_slime/train.jsonl
```

If you already have a local `hotpotqa_train.parquet`, pass `--input` instead of `--hf-dataset`.

**Eval set (RULER-HQA)** — `eval_{50,100,200,...}.json` files under `DATA_ROOT` (default `/data/datasets/hotpotqa_hf`):

```bash
mkdir -p /data/datasets/hotpotqa_hf

# Download all default lengths (50 … 6400)
bash examples/mem_agent/prepare-eval-data.sh --download

# Or only the lengths you need
LENGTHS="50 200 800" bash examples/mem_agent/prepare-eval-data.sh --download
```

To check which files are present without downloading:

```bash
LENGTHS="50 200 800" bash examples/mem_agent/prepare-eval-data.sh
```

#### Run pipeline

```bash
cd vime

# 1. Training (100 steps by default; set TRAIN_DATA if you used a different path)
bash examples/mem_agent/run-qwen3-4b-train.sh

# 2. Eval (after convert to HF)
CONVERT=1 SINGLE_ITER=iter_0000099 bash examples/mem_agent/run-eval.sh

# Baseline (untrained Qwen3-4B)
MODEL_PATH=/data/models/Qwen3-4B SAVE_FILE=Qwen3-4B-base bash examples/mem_agent/run-eval.sh
```

### Environment variables

Paths (override for your cluster):

- `HF_CKPT`, `TORCH_DIST`, `TRAIN_DATA`, `SAVE_PATH`, `DATA_ROOT`

Training:

- `NUM_ROLLOUT`, `MEM_CHUNK_TOKENS`, `MEM_MAX_MEMORY`, `MEM_MAX_FINAL`, `MEM_MAX_CHUNKS`

Eval:

- `MODEL_PATH`, `LENGTH` (e.g. `"50 200 800"`), `CONVERT`, `SINGLE_ITER`, `SAVE_FILE`

### Example paths

```bash
export HF_CKPT=/data/models/Qwen3-4B
export TORCH_DIST=/data/models/Qwen3-4B_torch_dist
export TRAIN_DATA=/data/datasets/hotpotqa_slime/train.jsonl
export SAVE_PATH=/data/models/MemAgent_Qwen3-4B-RL
export DATA_ROOT=/data/datasets/hotpotqa_hf

bash examples/mem_agent/run-qwen3-4b-train.sh
```
