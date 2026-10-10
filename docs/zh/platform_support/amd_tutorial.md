# AMD ROCm

本教程使用 ROCm 镜像。完整的启动脚本说明见[英文教程](https://github.com/vllm-project/vime/blob/main/docs/en/platform_support/amd_tutorial.md)。

## 环境准备

拉取镜像并启动容器：

```bash
docker pull vllm/vime-rocm
docker run --rm -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  vllm/vime-rocm /bin/bash
```

容器中安装当前 Vime checkout：

```bash
git clone https://github.com/vllm-project/vime.git /root/vime
cd /root/vime
pip install -e . --no-deps
```

也可以使用仓库中的 `docker/Dockerfile.rocm` 构建镜像。

## 模型与数据

```bash
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

## 权重转换

共享的转换脚本在 ROCm 环境下要求 `--use-cpu-initialization`，以便在 CPU 上初始化和保存模型权重。

```bash
cd /root/vime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --no-gradient-accumulation-fusion \
  --use-cpu-initialization \
  --attention-backend flash \
  --hf-checkpoint /root/Qwen3-8B \
  --save /root/Qwen3-8B_torch_dist
```

请将 `PYTHONPATH` 替换为镜像中实际的 Megatron-LM 安装路径。

## 训练

```bash
cd /root/vime
NUM_ROLLOUT=100 VISIBLE_GPUS=0,1 bash scripts/run-qwen3-8B-amd.sh
```

该配方通过 `--no-gradient-accumulation-fusion` 禁用梯度累积融合。Ray 的 GPU 可见性由 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES` 控制；调整启动脚本时请保留这些设置。
