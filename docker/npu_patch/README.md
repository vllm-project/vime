# Vime NPU Patch Installation Guide

This guide provides instructions for installing Vime with NPU support, including the required dependencies and patches.

## Component Version Mapping

| Component | Version/Commit | Source |
| --- | --- | --- |
| Base image | `v0.28.0-fd81546-a3` | `quay.io/atlas-ci/vllm-ascend` |
| vLLM | `e6bfe03ad73a3330cb427885aa90d97a12e1c704` | [GitHub](https://github.com/vllm-project/vllm) |
| vLLM-Ascend | `fd815467c221ee600137f6bdd53fe354d5e7c999` | [GitHub](https://github.com/vllm-project/vllm-ascend) |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| Megatron-Bridge | `3fd3768045422d0aa5c97e90a4e6c659aea9acb9` | [GitHub](https://github.com/radixark/Megatron-Bridge) |
| mbridge | `89eb10887887bc74853f89a4de258c0702932a1c` | [GitHub](https://github.com/ISEEKYAN/mbridge) |
| MegatronAdaptor | `15582addff3f3d4680e350826fa70d012b475509` | [GitCode](https://gitcode.com/Ascend/MegatronAdaptor) |
| TransformerEngineNPU | `d743c83d060d5edc48867ecb9e93ec80d81860e4` | [GitCode](https://gitcode.com/Ascend/TransformerEngineNPU) |
| MindSpeed | `fc63de5c48426dd019c3b3f39e65f5bdf56e4086` | [GitCode](https://gitcode.com/Ascend/MindSpeed) |
| torch_memory_saver (NPU) | `sgl-kernel-npu` tag `2026.6.0` | [GitHub](https://github.com/sgl-project/sgl-kernel-npu) |

## Preparing the Running Environment

Run the following steps inside the base image listed above, with the Ascend devices and host driver mounted. The base image provides Python, CANN, PyTorch, torch-npu, vLLM and vLLM-Ascend.

Use a checkout of this Vime revision at `/root/vime`. Start with unpatched dependency source trees; do not repeat these steps in an already-patched Vime image.

```bash
export VIME_INSTALL_ROOT=/root
export PATCH_DIR="${VIME_INSTALL_ROOT}/vime/docker/npu_patch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Preserve the base image's serving package versions when installing training dependencies:

```bash
export PIP_CONSTRAINT="$(mktemp /tmp/vime-npu-constraints.XXXXXX)"
python3 - <<'PY'
import importlib.metadata as metadata
import os

names = [
    "numpy", "ray", "torch", "torch-npu", "torchvision",
    "transformers", "triton-ascend", "vllm", "vllm-ascend",
]
with open(os.environ["PIP_CONSTRAINT"], "w") as constraints:
    constraints.write("\n".join(f"{name}=={metadata.version(name)}" for name in names) + "\n")
PY
```

### 1. vLLM and vLLM-Ascend

Both packages are installed in editable mode in the base image. Apply the patches to their existing source trees:

```bash
git -C /vllm-workspace/vllm apply --check "${PATCH_DIR}/vllm.patch"
git -C /vllm-workspace/vllm apply "${PATCH_DIR}/vllm.patch"

git -C /vllm-workspace/vllm-ascend apply --check "${PATCH_DIR}/vllm-ascend.patch"
git -C /vllm-workspace/vllm-ascend apply "${PATCH_DIR}/vllm-ascend.patch"
```

### 2. Megatron-LM

Apply the common Megatron patch before the NPU patch.

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git "${VIME_INSTALL_ROOT}/Megatron-LM"
git -C "${VIME_INSTALL_ROOT}/Megatron-LM" checkout 1dcf0dafa884ad52ffb243625717a3471643e087

git -C "${VIME_INSTALL_ROOT}/Megatron-LM" apply --check "${VIME_INSTALL_ROOT}/vime/docker/patch/latest/megatron.patch"
git -C "${VIME_INSTALL_ROOT}/Megatron-LM" apply "${VIME_INSTALL_ROOT}/vime/docker/patch/latest/megatron.patch"
git -C "${VIME_INSTALL_ROOT}/Megatron-LM" apply --check "${PATCH_DIR}/megatron.patch"
git -C "${VIME_INSTALL_ROOT}/Megatron-LM" apply "${PATCH_DIR}/megatron.patch"

pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/Megatron-LM"
```

### 3. Megatron-Bridge and mbridge

Use the Megatron-Bridge source through `PYTHONPATH`, without installing its CUDA package dependencies.

```bash
git clone --branch bridge https://github.com/radixark/Megatron-Bridge.git "${VIME_INSTALL_ROOT}/Megatron-Bridge"
git -C "${VIME_INSTALL_ROOT}/Megatron-Bridge" checkout 3fd3768045422d0aa5c97e90a4e6c659aea9acb9
git -C "${VIME_INSTALL_ROOT}/Megatron-Bridge" apply --check "${PATCH_DIR}/megatron-bridge.patch"
git -C "${VIME_INSTALL_ROOT}/Megatron-Bridge" apply "${PATCH_DIR}/megatron-bridge.patch"

git clone https://github.com/ISEEKYAN/mbridge.git "${VIME_INSTALL_ROOT}/mbridge"
git -C "${VIME_INSTALL_ROOT}/mbridge" checkout 89eb10887887bc74853f89a4de258c0702932a1c
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/mbridge"

pip install --no-build-isolation "nvidia-modelopt==0.46.0" "nvdlfw-inspect==0.2.2"
```

### 4. TransformerEngineNPU and MegatronAdaptor

Use TransformerEngineNPU, not the CUDA TransformerEngine package.

```bash
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git "${VIME_INSTALL_ROOT}/TransformerEngineNPU"
git -C "${VIME_INSTALL_ROOT}/TransformerEngineNPU" checkout d743c83d060d5edc48867ecb9e93ec80d81860e4
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/TransformerEngineNPU"

git clone https://gitcode.com/Ascend/MegatronAdaptor.git "${VIME_INSTALL_ROOT}/MegatronAdaptor"
git -C "${VIME_INSTALL_ROOT}/MegatronAdaptor" checkout 15582addff3f3d4680e350826fa70d012b475509
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/MegatronAdaptor"
```

### 5. MindSpeed

```bash
git clone https://gitcode.com/Ascend/MindSpeed.git "${VIME_INSTALL_ROOT}/MindSpeed"
git -C "${VIME_INSTALL_ROOT}/MindSpeed" checkout fc63de5c48426dd019c3b3f39e65f5bdf56e4086
git -C "${VIME_INSTALL_ROOT}/MindSpeed" apply --check "${PATCH_DIR}/mindspeed.patch"
git -C "${VIME_INSTALL_ROOT}/MindSpeed" apply "${PATCH_DIR}/mindspeed.patch"
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/MindSpeed"
```

### 6. Vime

```bash
pip install -r "${VIME_INSTALL_ROOT}/vime/requirements.txt"
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/vime"
```

### 7. torch_memory_saver

Build the NPU wheel from `sgl-kernel-npu`:

```bash
git clone --depth 1 --branch 2026.6.0 https://github.com/sgl-project/sgl-kernel-npu.git "${VIME_INSTALL_ROOT}/sgl-kernel-npu"
cd "${VIME_INSTALL_ROOT}/sgl-kernel-npu/contrib/torch_memory_saver/python"
python3 setup.py bdist_wheel
python3 -m pip install --no-deps dist/torch_memory_saver-*.whl
cd "${VIME_INSTALL_ROOT}/vime"
```

## Environment Setup and Installation Check

Set the source paths before running Vime:

```bash
export PYTHONPATH="${VIME_INSTALL_ROOT}/Megatron-Bridge/src:${VIME_INSTALL_ROOT}/Megatron-LM:${VIME_INSTALL_ROOT}/MegatronAdaptor:${VIME_INSTALL_ROOT}/TransformerEngineNPU:${VIME_INSTALL_ROOT}/vime${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c 'import megatron, mindspeed, megatron_adaptor, transformer_engine, torch_memory_saver, vime, vllm, vllm_ascend'
```

For a complete container build recipe, see [Dockerfile.npu](../Dockerfile.npu).
