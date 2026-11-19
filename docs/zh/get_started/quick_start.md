# 运行环境准备

从 {ref}`交互 Quick Start <lab>` 开始。本页准备所有生成配方共用的环境；[tutorial](experiment-guide.md)讲解完整 RL 流程。

## 基础环境搭建

建议使用预置的 Docker 镜像，获得相互兼容的 vLLM、Megatron 和补丁环境。

### 硬件与镜像选择

| 硬件 | 镜像 | CUDA |
| --- | --- | --- |
| H100/H200 | `vllm/vime:latest` | 13.0 |
| B200/B300（Blackwell，x86） | `vllm/vime:latest` | 13.0 |

新构建的 Hopper 和 Blackwell 环境统一使用 CUDA 13 镜像。在受支持的数据中心 GPU 上，镜像会为 535、570 等旧驱动自动启用 CUDA 13 向前兼容库，不替换宿主机驱动。构建选项及驱动兼容说明见 [Docker 指南](https://github.com/vllm-project/vime/blob/main/docker/README.md)。

GPU CI 主要覆盖 H100/H200。使用其他平台时，请先核对具体训练配方的硬件和并行配置要求。

AMD 支持见 [AMD 使用教程](../platform_support/amd_tutorial.md)。

### 拉取并启动 Docker 容器

在 Linux x86_64 GPU 主机上准备 Git、Docker、兼容的 NVIDIA 驱动，并配置好 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。在**宿主机上**执行以下命令，拉取镜像、克隆 vime 并启动容器：

```shell
VIME_IMAGE=vllm/vime:latest
docker pull "$VIME_IMAGE"
git clone https://github.com/vllm-project/vime.git
cd vime
mkdir -p ../vime-data
docker run --rm -it --gpus all --ipc=host \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/root/vime" -v "$PWD/../vime-data:/data" \
  -w /root/vime "$VIME_IMAGE" /bin/bash
```

### 安装 vime

镜像已经包含 vime 及所需依赖；挂载后，镜像内的代码目录被宿主机 checkout 替代。在**容器内**执行以下命令，安装该 checkout、保留镜像中固定的依赖版本，并检查模块导入和 GPU 可见性：

```bash
pip install -e . --no-deps
python -c "import vime, vllm, megatron.core; import torch; print('GPUs:', torch.cuda.device_count())"
```

输出的 GPU 数量应与容器可见设备一致。将配置器导出的 `experiment.sh` 保存到宿主机的 `vime` 目录后，即可在容器内的 `/root/vime/experiment.sh` 访问它。容器内 `/data` 下的模型、数据及 checkpoint 会保存在宿主机旁边的 `vime-data` 目录，退出容器后仍然保留。再次进入环境时，在同一 checkout 下重新执行 `docker run` 命令，并在新容器内执行上述 editable install。

## 挂载实验路径

上述 Docker 命令已经将持久化数据目录挂载到 `/data`，与配置器默认值一致。使用已有数据或共享存储时，可将该挂载替换为 `-v /shared/data:/data`，并在向导中填写容器内可见路径。所有 Ray 节点以相同路径访问仓库和生成配置目录，安装相同依赖，并配置节点之间可达的网络；上述单机 Docker 命令不会自动搭好多机网络。

继续阅读 [tutorial 的准备与启动步骤](experiment-guide.md)，完成所选模型的下载、转换和训练。已有 launcher 的参数说明见[使用参考](usage.md)。
