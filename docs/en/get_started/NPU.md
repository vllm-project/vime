# NPU

⚠️ If you encounter problems running vime on Ascend NPU, feel free to open an
issue on [vllm-project/vime](https://github.com/vllm-project/vime/issues).

## Introduction

If you are running vime on Ascend NPU, please refer to the following materials.
This tutorial explains how to set up the runtime environment and provides an
end-to-end example for running GRPO training. It uses the **Megatron** training
backend together with the **vLLM Ascend** rollout backend, synchronizing actor 
weights to vLLM through the native HCCL weight-sync path.

The current NPU support targets Ascend **Atlas A2 / A3** (aarch64) hosts with the
Ascend driver and **CANN 9.0.0** (Toolkit, Kernels, and NNAL/ATB) installed.
Only `python==3.12` is supported.

## Docker

The recommended path for validation is the published vime NPU image.

```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2:  export IMAGE=quay.io/ascend/vime:vime-a2-latest

docker pull "${IMAGE}"
```

For source builds and dependency debugging, the patch list and pinned commits are
documented in [`docker/npu_patch/README.md`](https://github.com/vllm-project/vime/blob/npu/docker/npu_patch/README.md).

## Quick Start

### Environment Setup

Start the container, mounting the Ascend devices and driver files. Device names
and driver mount paths vary by host; reuse the mounts from a known working vLLM
Ascend container if the layout differs.

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

### Prepare Model and Data

Set `MODEL_ROOT` to a host-visible directory that will hold both the checkpoint
and the dataset, then download the Qwen3-4B checkpoint and the DAPO Math 17K
dataset:

```bash
export MODEL_ROOT=/root
mkdir -p ${MODEL_ROOT}/models ${MODEL_ROOT}/datasets

# hf checkpoint
hf download Qwen/Qwen3-4B \
  --local-dir ${MODEL_ROOT}/models/Qwen3-4B

# train data
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir ${MODEL_ROOT}/datasets/dapo-math-17k
```

### Example: Qwen3-4B

We provide an example to run GRPO training with
[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on 8 NPUs (4 for the actor,
4 for rollout), please refer to:
[scripts/models/qwen3-4B_npu.sh](https://github.com/vllm-project/vime/blob/npu/scripts/models/qwen3-4B_npu.sh).
Just run:

```bash
cd /root/vime

# Source these explicitly if not already initialized by the image.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_ROOT=/root bash scripts/models/qwen3-4B_npu.sh
```

The full log is written to `/root/vime/train_qwen3_4b_vllm.log`.

⚠️ Note: The main difference between the NPU training script and the NVIDIA one
is the Ascend-specific environment variables — `ASCEND_RT_VISIBLE_DEVICES`
selects the NPUs, and `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` lets
Ray schedule them correctly. The reference target is an Atlas A3 host with 16
visible NPUs; on an 8-NPU host, set
`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

We show the training script below:

```bash
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)  # or any free port
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0

SCRIPT_DIR="/root/vime/scripts/"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
LOG_FILE="/root/vime/train_qwen3_4b_vllm.log"
MODEL_ROOT="${MODEL_ROOT:-/root}"

python /root/vime/train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  ${MODEL_ARGS[@]} \
  \
  --hf-checkpoint ${MODEL_ROOT}/models/Qwen3-4B/ \
  \
  --prompt-data ${MODEL_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  \
  --rollout-backend vllm \
  --vllm-weight-sync-mode native \
  --vllm-gpu-memory-utilization 0.6 \
  --vllm-enable-sleep-mode \
  --vllm-max-model-len 4096 \
  \
  --num-rollout 200 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 2048 \
  --rollout-temperature 1.0 \
  --global-batch-size 256 \
  --balance-data \
  \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.00 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --load ${MODEL_ROOT}/models/Qwen3-4B \
  --megatron-to-hf-mode bridge \
  \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --micro-batch-size 1 \
  --use-flash-attn \
  \
  --train-memory-margin-bytes 2147483648 \
  2>&1 | tee -a "$LOG_FILE"
```
