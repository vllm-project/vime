# Installing FlashQLA

FlashQLA is an optional Gated Delta Net (GDN) backend for Qwen3-Next and Qwen3.5. To select it, add this argument to the training command:

```bash
--qwen-gdn-backend flashqla
```

The default backend is FLA.

## Requirements

- PyTorch 2.8 or newer.
- CUDA 12.8 or newer.
- NVIDIA SM90 or newer GPUs.
- The same FlashQLA installation on every training node.

## Docker Images

Vime omits FlashQLA by default. To request its installation, use the existing Docker build option:

```bash
docker build -f docker/Dockerfile . \
  --build-arg INSTALL_FLASHQLA=1 \
  -t vime:flashqla
```

FlashQLA currently requires TileLang 0.1.9. Do not downgrade a serving environment's incompatible TileLang dependency; use the default FLA backend or a separate compatible training environment. Validate compilation and runtime behavior before using FlashQLA in training.
