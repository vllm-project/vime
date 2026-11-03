# Vime NPU Patch Installation Guide

This guide provides instructions for installing Vime with NPU support, including the required dependencies and patches.

## Component Version Mapping

| Component | Version/Commit | Source |
| --- | --- | --- |
| vLLM | `544f182d9f0c9bfe8bd7645525951d5e3692a542` | [GitHub](https://github.com/vllm-project/vllm) |
| vLLM-Ascend | `2d6654fbd170d110eec9c6cb30c55751045c4d8d` | [GitHub](https://github.com/vllm-project/vllm-ascend) |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| MegatronAdaptor | `15582addff3f3d4680e350826fa70d012b475509` | [GitCode](https://gitcode.com/Ascend/MegatronAdaptor) |
| TransformerEngineNPU | `d743c83d060d5edc48867ecb9e93ec80d81860e4` | [GitCode](https://gitcode.com/Ascend/TransformerEngineNPU) |
| MindSpeed | `fc63de5c48426dd019c3b3f39e65f5bdf56e4086` | [GitCode](https://gitcode.com/Ascend/MindSpeed) |
| torch_memory_saver (NPU) | `sgl-kernel-npu` tag `2026.6.0` | [GitHub](https://github.com/sgl-project/sgl-kernel-npu) |

## Preparing the Running Environment

This patch series targets the vLLM and vLLM-Ascend source commits above.
The Dockerfile and its base image are unchanged. Its default v0.28 image does
not contain these commits and cannot apply this reduced series as-is. Rebuild
or update the serving sources separately before using this series in an image;
image migration is outside this source-only change.

Ascend now supplies the HCCL trainer registration/implementation, IPC layerwise
reload lifecycle, packed IPC loading, KV-cache allocation context and
kernel-block slot mapping. Those backports have been removed. The remaining
patch retains draft/rank-local updates, IPC memory and pack lifecycle handling,
disk checkpoint updates and model adaptations. Existing Vime response fields
remain available alongside the native choice-level version spans and metrics.

Run the following steps in a source environment using the commits above, with
compatible CANN, PyTorch and torch-npu installed and Ascend devices mounted.
Local source-level and CPU checks do not establish end-to-end NPU compatibility.

本补丁仅针对上表的 vLLM / vLLM-Ascend 源码版本，未修改镜像或 Dockerfile。
旧版 v0.28 镜像不能直接应用精简后的补丁；镜像适配需要单独处理。

To check the series against local source repositories without modifying them:

```bash
VLLM_SOURCE_DIR=/path/to/vllm \
VLLM_ASCEND_SOURCE_DIR=/path/to/vllm-ascend \
python -m unittest discover -s tests -p test_npu_weight_patches.py -v
```

This copies the affected files from each repository's `HEAD`, checks/applies the
patches in temporary directories, parses the resulting Python files, and runs
isolated CPU regressions for rank routing, draft cleanup, empty payloads, IPC
collection, packed references/stream ordering and response compatibility.
CPU PyTorch is needed for the packing tests. No NPU runtime is exercised.

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
    # PyArrow 26 requires NumPy 2, while the NPU Triton stack uses NumPy 1.26.4.
    constraints.write("pyarrow==21.0.0\n")
    constraints.write("datasets==4.4.2\n")
PY
```

PyArrow 21.0.0 with datasets 4.4.2 supports the image's NumPy 1.26.4 and Parquet rollout datasets.
Keep this constraint when installing Vime dependencies; upgrading NumPy to satisfy
PyArrow 26 conflicts with the image's Triton requirements. datasets 5.1 requires
PyArrow >=24, so it must be constrained together with PyArrow.

### 1. vLLM and vLLM-Ascend

Apply the patches to source trees at the pinned commits above. If serving packages
are installed in editable mode, these edits take effect through that installation.
If they were installed as ordinary wheels, rebuild and reinstall both packages
from the patched trees before testing; changing the checkout alone is insufficient.
Check the imported package paths and run a Vime parser/server smoke test.

The pinned vLLM uses `vllm.entrypoints.common.cli_args`. Vime must use that
module for frontend CLI imports. Its rollout also requires the scale-out
endpoint: keep `vllm_enable_scale_out=True` for Vime-launched servers, or
supply `--enable-scale-out` when launching an external vLLM server.

Apply the source patches:

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

### 3. TransformerEngineNPU and MegatronAdaptor

Use TransformerEngineNPU, not the CUDA TransformerEngine package.

```bash
pip install --no-build-isolation "nvidia-modelopt==0.46.0" "nvdlfw-inspect==0.2.2"

git clone https://gitcode.com/Ascend/TransformerEngineNPU.git "${VIME_INSTALL_ROOT}/TransformerEngineNPU"
git -C "${VIME_INSTALL_ROOT}/TransformerEngineNPU" checkout d743c83d060d5edc48867ecb9e93ec80d81860e4
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/TransformerEngineNPU"

git clone https://gitcode.com/Ascend/MegatronAdaptor.git "${VIME_INSTALL_ROOT}/MegatronAdaptor"
git -C "${VIME_INSTALL_ROOT}/MegatronAdaptor" checkout 15582addff3f3d4680e350826fa70d012b475509
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/MegatronAdaptor"
```

### 4. MindSpeed

```bash
git clone https://gitcode.com/Ascend/MindSpeed.git "${VIME_INSTALL_ROOT}/MindSpeed"
git -C "${VIME_INSTALL_ROOT}/MindSpeed" checkout fc63de5c48426dd019c3b3f39e65f5bdf56e4086
git -C "${VIME_INSTALL_ROOT}/MindSpeed" apply --check "${PATCH_DIR}/mindspeed.patch"
git -C "${VIME_INSTALL_ROOT}/MindSpeed" apply "${PATCH_DIR}/mindspeed.patch"
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/MindSpeed"
```

### 5. Vime

```bash
pip install -r "${VIME_INSTALL_ROOT}/vime/requirements.txt"
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/vime"
```

### 6. torch_memory_saver

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
export PYTHONPATH="${VIME_INSTALL_ROOT}/Megatron-LM:${VIME_INSTALL_ROOT}/MegatronAdaptor:${VIME_INSTALL_ROOT}/TransformerEngineNPU:${VIME_INSTALL_ROOT}/vime${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c 'import megatron, mindspeed, megatron_adaptor, transformer_engine, torch_memory_saver, vime, vllm, vllm_ascend'
```

For a complete container build recipe, see [Dockerfile.npu](../Dockerfile.npu).
