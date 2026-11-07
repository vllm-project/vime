# 运行环境准备

从 {ref}`交互 Quick Start <lab>` 开始。本页准备所有生成配方共用的环境；[tutorial](experiment-guide.md)讲解完整 RL 流程。

## 基础环境搭建

建议使用预置的 Docker 镜像，获得相互兼容的 vLLM、Megatron 和补丁环境。

### 硬件与镜像选择

| 硬件 | 镜像 | CUDA |
| --- | --- | --- |
| H100/H200 | `vllm/vime:latest` | 13.0 |
| B200/B300（Blackwell，x86） | `vllm/vime:latest` | 13.0 |

Vime 对这两类硬件使用 CUDA 13 镜像；依赖版本和构建选项见 [Docker 指南](https://github.com/vllm-project/vime/blob/main/docker/README.md)。

GPU CI 主要覆盖 H100/H200。使用其他平台时，请先核对具体训练配方的硬件和并行配置要求。

AMD 支持见 [AMD 使用教程](../platform_support/amd_tutorial.md)。

### 拉取并启动 Docker 容器

请执行以下命令，拉取最新镜像并启动一个交互式容器：

```shell
# 拉取最新镜像
docker pull vllm/vime:latest

# 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it vllm/vime:latest /bin/bash
```

### 安装 vime

vime 已经安装在该 Docker 镜像中。如需更新到最新版本，请在 Docker 容器中执行以下命令：

```bash
# 路径可根据实际情况调整
cd /root/vime
git pull
pip install -e . --no-deps
```

## 挂载实验路径

启动容器时挂载数据目录，例如 `-v /shared/data:/data`，并在向导中填写容器内可见路径。所有 Ray 节点以相同路径访问仓库和生成配置目录，并安装相同依赖。

继续阅读 [tutorial 的准备与启动步骤](experiment-guide.md)，完成所选模型的下载、转换和训练。已有 launcher 的参数说明见[使用参考](usage.md)。
