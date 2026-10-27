# NPU

⚠️ 如果在 Ascend NPU 上运行 vime 时遇到问题，欢迎在
[vllm-project/vime](https://github.com/vllm-project/vime/issues) 提交 issue。

## 简介

本教程介绍在 Ascend NPU 上运行 vime 所需的环境配置，并提供完整的 GRPO 训练示例。
示例使用 **Megatron** 训练后端和 **vLLM Ascend** rollout 后端，通过原生 HCCL
权重同步路径将 actor 权重同步到 vLLM。

当前 NPU 支持面向 Ascend **Atlas A2 / A3**（aarch64）主机，需要安装 Ascend 驱动和
**CANN 9.1.0**（Toolkit、Kernels 和 NNAL/ATB）。仅支持 `python==3.12`。

## Docker

建议使用已发布的 vime NPU 镜像进行验证。

```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2:  export IMAGE=quay.io/ascend/vime:vime-a2-latest

docker pull "${IMAGE}"
```

如需从源码构建或调试依赖，请参阅
[`docker/npu_patch/README.md`](https://github.com/vllm-project/vime/blob/main/docker/npu_patch/README.md)
中的补丁清单和固定提交版本。

## 快速开始

### 环境准备

启动容器并挂载 Ascend 设备和驱动文件。不同主机的设备名称和驱动路径可能不同，
请根据实际环境调整；也可以沿用该主机上已经验证可用的 vLLM Ascend 容器挂载配置。

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

进入容器后，在训练前初始化 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 准备模型和数据

将 `MODEL_ROOT` 设置为存放模型和数据集的目录，并确保该目录通过容器挂载与主机共享。
然后下载 Qwen3-4B 模型和 DAPO Math 17K 数据集：

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

### 示例：Qwen3-4B

仓库提供了在 8 张 NPU 上使用 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
进行 GRPO 训练的示例，其中 4 张用于 actor，4 张用于 rollout。启动脚本为
[scripts/run-qwen3-4B-npu.sh](../../../scripts/run-qwen3-4B-npu.sh)。执行：

```bash
cd /root/vime

# Source these explicitly if not already initialized by the image.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

DATA_ROOT="${MODEL_ROOT:-/root}" bash scripts/run-qwen3-4B-npu.sh \
  2>&1 | tee /root/vime/train_qwen3_4b_vllm.log
```

完整日志保存在 `/root/vime/train_qwen3_4b_vllm.log`。

⚠️ 注意：NPU 训练脚本与 NVIDIA 版本的主要区别是 Ascend 专用环境变量。
`ASCEND_RT_VISIBLE_DEVICES` 用于指定可见的 NPU，
`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` 用于配合 Ray 的设备调度。
参考环境是具有 16 张可见 NPU 的 Atlas A3 主机；如果使用 8 卡主机，请设置
`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

训练脚本示例如下：

```bash
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/root/Megatron-LM/:/root/MegatronAdaptor:/root/TransformerEngineNPU:$PYTHONPATH"
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
