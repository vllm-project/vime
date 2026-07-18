# Quick Start

This document will guide you through setting up the environment and getting started with vime on AMD ROCm, covering environment configuration, data preparation, weight conversion, and training startup.

## Basic Environment Setup

### Pull and Start Docker Container

Execute the following commands to pull the latest ROCm image and start a persistent container:

```shell
# Pull the ROCm image
docker pull vllm/vime-rocm

# Start the container
docker run -d --name vime --ulimit nofile=1048576:1048576 \
  --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video --privileged \
  -e WANDB_API_KEY=$wandb_key vllm/vime-rocm
# wandb key is optional if you want to track with WandB

# Enter the container
docker exec -it vime bash
```

## Model and Dataset Download

Download the required model and training dataset using `huggingface_hub`:

```bash
# Download model weights (Qwen3-8B)
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B

# Download training dataset (dapo-math-17k)
hf download zhuzilin/dapo-math-17k --repo-type dataset --local-dir /root/dapo-math-17k
```

## Model Weight Conversion

### Convert from Hugging Face Format to Megatron Format

Load the model configuration for Qwen3-8B, then run the conversion. Two ROCm-specific flags are required: `--no-gradient-accumulation-fusion` and `--attention-backend flash`.

```bash
cd /root/vime && source scripts/models/qwen3-8B.sh

HIP_VISIBLE_DEVICES=0 PYTHONPATH=/root/vime:/root/Megatron-LM \
  torchrun --nproc-per-node=1 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion --attention-backend flash \
  --hf-checkpoint /root/Qwen3-8B --save /root/Qwen3-8B_torch_dist
```

> **Note**: On ROCm, use `HIP_VISIBLE_DEVICES` in place of `CUDA_VISIBLE_DEVICES` to select GPUs. The `--attention-backend flash` and `--no-gradient-accumulation-fusion` flags are required to avoid issues during conversion.

## Training

Run training using the ROCm-specific script. `VISIBLE_GPUS` specifies which GPUs to use, and `NUM_ROLLOUT` controls the total number of sampling→training rounds. The script automatically unsets NVTE environment variables that are not needed on ROCm.

```bash
NUM_ROLLOUT=100 VISIBLE_GPUS=0,1 bash scripts/run-qwen3-8B-amd.sh
```

### Configuration Notes

- **VISIBLE_GPUS**
    - Two free GPU indices.
    - Script masks execution to these GPUs, avoids clashes.
    - Uses:
        - TP = 2
        - Single vLLM engine
        - Colocate mode
        - DP = 1
- **NUM_ROLLOUT**
    - Number of training steps.
    - Default is **3** for a smoke test.
- Each run requires approximately **230 GB** across the two selected GPUs.
- Launch only on GPUs with sufficient free memory.

> **Final note**: After finishing the run, if rerunning with a different `NUM_ROLLOUT`, make sure to clear the save directory to avoid mismatch error.
```bash
rm -rf /root/Qwen3-8B_vime/
```