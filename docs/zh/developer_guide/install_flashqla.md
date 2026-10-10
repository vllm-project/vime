# 安装 FlashQLA

FlashQLA 是 Qwen3-Next 和 Qwen3.5 的可选 Gated Delta Net（GDN）后端。安装后仍需要在训练命令中显式加入：

```bash
--qwen-gdn-backend flashqla
```

默认后端是 FLA。

## 环境要求

- PyTorch 2.8 或更新版本。
- CUDA 12.8 或更新版本。
- NVIDIA SM90 或更新架构 GPU。
- 所有训练节点都安装相同的 FlashQLA Python 包。

## Docker 镜像

Vime 默认不安装 FlashQLA。需要安装时，使用现有 Docker 构建选项：

```bash
docker build -f docker/Dockerfile . \
  --build-arg INSTALL_FLASHQLA=1 \
  -t vime:flashqla
```

FlashQLA 当前要求 TileLang 0.1.9。不要降级 serving 环境中与之不兼容的 TileLang 依赖；请使用默认 FLA 后端，或独立的兼容训练环境。用于训练前需要验证编译和运行行为。
