# Ascend NPU 快速上手

> **分支说明：** Ascend NPU 支持目前维护在 [ascend](https://github.com/vllm-project/vime/tree/ascend)
> 分支（尚未合入 `main`），后续有计划将其合并至 `main`。
> 运行下文任何 NPU 示例前，请先 clone 或 checkout 该分支。

⚠️ 如在 Ascend NPU 上运行 vime 遇到问题，欢迎在
[vllm-project/vime](https://github.com/vllm-project/vime/issues) 提交 Issue。

## 概述

vime 在 Ascend NPU 上使用 **Megatron** 训练后端与 **vLLM Ascend** rollout 后端。
解耦模式下 actor 权重经 HCCL 同步到 vLLM；colocate 模式（`--colocate`）下经 NPU IPC 同步。

当前支持 Ascend **Atlas A2 / A3**（aarch64）硬件。

## 获取 ascend 分支

```bash
git clone --branch ascend https://github.com/vllm-project/vime.git
cd vime
```

若已有仓库：

```bash
git fetch origin ascend
git checkout ascend
```

## ascend 分支资源索引

| 资源 | 说明 |
| ---- | ---- |
| [docs/en/get_started/NPU.md](https://github.com/vllm-project/vime/blob/ascend/docs/en/get_started/NPU.md) | 完整 NPU 指南，含 GRPO 端到端示例与训练参数 |
| [docker/npu_patch/README.md](https://github.com/vllm-project/vime/blob/ascend/docker/npu_patch/README.md) | 源码构建、依赖版本与 patch 列表 |
| [scripts/run-qwen3-4B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/run-qwen3-4B-npu.sh) | Qwen3-4B 解耦训练（4 actor + 4 rollout NPU） |
| [scripts/run-qwen3-30B-A3B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/run-qwen3-30B-A3B-npu.sh) | Qwen3-30B-A3B MoE NPU 训练脚本 |
| [scripts/models/qwen3-30B-A3B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/models/qwen3-30B-A3B-npu.sh) | Qwen3-30B-A3B NPU 模型参数 |

## 基础环境

### Docker 镜像

推荐使用已发布的 vime NPU 镜像：

```bash
export IMAGE=quay.io/ascend/vime:vime-latest
# A2: export IMAGE=quay.io/ascend/vime:vime-a2-latest

docker pull "${IMAGE}"
```

源码构建与依赖调试请参考 `ascend` 分支上的
[docker/npu_patch/README.md](https://github.com/vllm-project/vime/blob/ascend/docker/npu_patch/README.md)。

### 拉取并启动容器

挂载 Ascend 设备与驱动文件后启动容器。设备名与挂载路径因主机而异，可参考已跑通的 vLLM Ascend 容器配置。

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

容器内训练前初始化 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 模型与数据集下载

```bash
export MODEL_ROOT=/root
mkdir -p ${MODEL_ROOT}/models ${MODEL_ROOT}/datasets

# 模型权重（Qwen3-4B）
hf download Qwen/Qwen3-4B --local-dir ${MODEL_ROOT}/models/Qwen3-4B

# 训练数据集（dapo-math-17k）
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir ${MODEL_ROOT}/datasets/dapo-math-17k
```

## 训练示例（Qwen3-4B）

在容器内 checkout `ascend` 分支后，运行脚本：

```bash
cd /root/vime

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

MODEL_ROOT=/root bash scripts/run-qwen3-4B-npu.sh
```

完整日志写入 `/root/vime/train_qwen3_4b_vllm.log`。

> **说明：** 与 NVIDIA 流程的主要区别是 Ascend 环境变量 — 使用
> `ASCEND_RT_VISIBLE_DEVICES` 替代 `CUDA_VISIBLE_DEVICES`，并设置
> `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` 以便 Ray 正确调度 NPU。
> 参考脚本面向 16 卡 Atlas A3；8 卡主机请设置
> `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

完整训练命令、HCCL 端口范围与参数说明见
[ascend 分支 NPU.md](https://github.com/vllm-project/vime/blob/ascend/docs/en/get_started/NPU.md)。

## MoE 示例（Qwen3-30B-A3B）

MoE 模型请使用 `ascend` 分支脚本：

```bash
bash scripts/run-qwen3-30B-A3B-npu.sh
```

模型参数见
[scripts/models/qwen3-30B-A3B-npu.sh](https://github.com/vllm-project/vime/blob/ascend/scripts/models/qwen3-30B-A3B-npu.sh)。
