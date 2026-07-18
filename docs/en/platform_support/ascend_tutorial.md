# Ascend NPU Quick Start

> **Branch notice:** Ascend NPU support is currently maintained on the [ascend](https://github.com/vllm-project/vime/tree/ascend)
> branch (not yet on `main`), with plans to merge into `main` later.
> Clone or checkout that branch before running any NPU examples below.

⚠️ If you encounter problems running vime on Ascend NPU, feel free to open an
issue on [vllm-project/vime](https://github.com/vllm-project/vime/issues).

## Overview

vime on Ascend NPU uses the **Megatron** training backend together with the
**vLLM Ascend** rollout backend. In decoupled mode, actor weights sync to vLLM
over HCCL; in colocate mode (`--colocate`), weights sync over NPU IPC.

Current support targets Ascend **Atlas A2 / A3** (aarch64) hardware.

## Get the Ascend Branch

```bash
git clone --branch ascend https://github.com/vllm-project/vime.git
cd vime
```

If you already have the repo:

```bash
git fetch origin ascend
git checkout ascend
```

## Ascend Branch Resources

| Resource | Description |
| -------- | ----------- |
| [docs/en/get_started/NPU.md](https://github.com/vllm-project/vime/blob/ascend/docs/en/get_started/NPU.md) | Full NPU guide with end-to-end GRPO example and training flags |
| [docker/npu_patch/README.md](https://github.com/vllm-project/vime/blob/ascend/docker/npu_patch/README.md) | Source-build guide, pinned commits, and patch list |
| [scripts/run-qwen3-4B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/run-qwen3-4B-npu.sh) | Qwen3-4B decoupled training (4 actor + 4 rollout NPUs) |
| [scripts/run-qwen3-30B-A3B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/run-qwen3-30B-A3B-npu.sh) | Qwen3-30B-A3B MoE NPU training script |
| [scripts/models/qwen3-30B-A3B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/models/qwen3-30B-A3B-npu.sh) | Model args for Qwen3-30B-A3B on NPU |

## Basic Environment Setup

### Docker Image

The recommended path for validation is the published vime NPU image:

```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2: export IMAGE=quay.io/ascend/vime:vime-a2-latest

docker pull "${IMAGE}"
```

For source builds and dependency debugging, follow
[docker/npu_patch/README.md](https://github.com/vllm-project/vime/blob/ascend/docker/npu_patch/README.md)
on the `ascend` branch.

### Pull and Start Docker Container

Start the container with Ascend devices and driver files mounted. Device names
and mount paths vary by host; reuse the mounts from a known working vLLM Ascend
container if the layout differs.

```bash
docker run -d --name vime-npu -it --net=host --shm-size=1024g \
    --privileged=true \
    --cap-add=SYS_PTRACE \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /path/to:/path/to \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    "${IMAGE}"

docker exec -it vime-npu bash
```

Inside the container, initialize the CANN environment before training:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## Model and Dataset Download

```bash
export MODEL_ROOT=/root
mkdir -p ${MODEL_ROOT}/models ${MODEL_ROOT}/datasets

# Model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir ${MODEL_ROOT}/models/Qwen3-4B

# Training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir ${MODEL_ROOT}/datasets/dapo-math-17k
```

## Training (Qwen3-4B Example)

After checking out the `ascend` branch inside the container, run the bundled
script:

```bash
cd /root/vime

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_ROOT=/root bash scripts/run-qwen3-4B-npu.sh
```

The full log is written to `/root/vime/train_qwen3_4b_vllm.log`.

> **Note:** The main difference from the NVIDIA workflow is Ascend-specific
> environment variables — use `ASCEND_RT_VISIBLE_DEVICES` instead of
> `CUDA_VISIBLE_DEVICES`, and set
> `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` so Ray schedules NPUs
> correctly. The reference script targets an Atlas A3 host with 16 visible NPUs;
> on an 8-NPU host, set `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

For the full training command, HCCL port ranges, and flag explanations, see
[NPU.md on the ascend branch](https://github.com/vllm-project/vime/blob/ascend/docs/en/get_started/NPU.md).

## MoE Example (Qwen3-30B-A3B)

For the MoE model on NPU, use the scripts on the `ascend` branch:

```bash
bash scripts/run-qwen3-30B-A3B-npu.sh
```

See [scripts/models/qwen3-30B-A3B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/models/qwen3-30B-A3B-npu.sh)
for model-specific arguments.
