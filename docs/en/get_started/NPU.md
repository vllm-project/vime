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
Ascend driver and **CANN 9.1.0** (Toolkit, Kernels, and NNAL/ATB) installed.
Only `python==3.12` is supported.

## Docker

The recommended path for validation is the published vime NPU image.

```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2:  export IMAGE=quay.io/ascend/vime:vime-a2-latest

docker pull "${IMAGE}"
```

For source builds and dependency debugging, the patch list and pinned commits are
documented in [`docker/npu_patch/README.md`](https://github.com/vllm-project/vime/blob/ascend/docker/npu_patch/README.md).

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
[scripts/run-qwen3-4B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/run-qwen3-4B-npu.sh).
Just run:

```bash
cd /root/vime

# Source these explicitly if not already initialized by the image.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

DATA_ROOT="${MODEL_ROOT:-/root}" bash scripts/run-qwen3-4B-npu.sh \
  2>&1 | tee /root/vime/train_qwen3_4b_vllm.log
```

The full log is written to `/root/vime/train_qwen3_4b_vllm.log`.

The script sets `ASCEND_RT_VISIBLE_DEVICES` and starts Ray with eight NPU
resources. Adjust the device list in the script to match the NPUs visible
inside your container. The script passes the training and vLLM arguments to
`train.py` through Ray.

To verify the first HCCL weight sync, add `--check-weight-update-equal` to the
training arguments. See [Check the initial weight update](../developer_guide/debug.md#check-the-initial-weight-update)
for the required serving patches and how to read the result.
