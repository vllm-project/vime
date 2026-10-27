# Vime NPU Patch Installation Guide

This guide provides instructions for installing Vime with NPU support, including the required dependencies and patches.

## Component Version Mapping

| Component | Version/Commit | Source |
| --- | --- | --- |
| Base image | `v0.31.0-007abb8-a3` | `quay.io/atlas-ci/vllm-ascend` |
| vLLM | `a98247ab4db686ee03c66d5feb3c761e52a2f8ab` | [GitHub](https://github.com/vllm-project/vllm) |
| vLLM-Ascend | `007abb8c89971f589932584a5a6d0a986bfbcdd0` | [GitHub](https://github.com/vllm-project/vllm-ascend) |
| vLLM Router | `1fbcde7443d75b36befb61bc081f64c2a1f13a4b` + `docker/patch/latest/router-health-check.patch` | [GitHub](https://github.com/vllm-project/router) |
| Sunabako | `0.1.1` + `docker/patch/latest/sunabako-arm64.patch` | [PyPI](https://pypi.org/project/sunabako/0.1.1/) |
| FLA (serving, included in base image) | `flash-linear-attention-npu-a3==26.9.1+dev5cc6990` | [v26.9.1-20261020-020004](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.9.1-20261020-020004) |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| MegatronAdaptor | `15582addff3f3d4680e350826fa70d012b475509` | [GitCode](https://gitcode.com/Ascend/MegatronAdaptor) |
| TransformerEngineNPU | `d743c83d060d5edc48867ecb9e93ec80d81860e4` | [GitCode](https://gitcode.com/Ascend/TransformerEngineNPU) |
| MindSpeed | `fc63de5c48426dd019c3b3f39e65f5bdf56e4086` | [GitCode](https://gitcode.com/Ascend/MindSpeed) |
| torch_memory_saver (NPU) | `sgl-kernel-npu` tag `2026.6.0` + `torch-memory-saver.patch` | [GitHub](https://github.com/sgl-project/sgl-kernel-npu) |

## Preparing the Running Environment

Run the following steps inside the base image listed above, with the Ascend devices and host driver mounted. The base image provides Python, CANN, PyTorch, torch-npu, vLLM, vLLM-Ascend and serving FLA.

Use a checkout of this Vime revision at `/root/vime`. Start with unpatched dependency source trees; do not repeat these steps in an already-patched Vime image.

```bash
export VIME_INSTALL_ROOT=/root
export PATCH_DIR="${VIME_INSTALL_ROOT}/vime/docker/npu_patch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Preserve the base image's serving package versions and pin the data dependencies compatible with NumPy 1.26.4:

```bash
export PIP_CONSTRAINT="$(mktemp /tmp/vime-npu-constraints.XXXXXX)"
python3 - <<'PY'
import importlib.metadata as metadata
import os

names = [
    "numpy", "ray", "torch", "torch-npu", "torchvision",
    "transformers", "triton-ascend", "vllm", "vllm-ascend",
    "fastapi", "starlette",
]
with open(os.environ["PIP_CONSTRAINT"], "w") as constraints:
    constraints.write("\n".join(f"{name}=={metadata.version(name)}" for name in names) + "\n")
    constraints.write("datasets==5.0.1\npyarrow==25.0.1\n")
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

Apply the single NPU patch directly to the pinned commit. Do not apply the public
`docker/patch/latest/megatron.patch` first.

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git "${VIME_INSTALL_ROOT}/Megatron-LM"
git -C "${VIME_INSTALL_ROOT}/Megatron-LM" checkout 1dcf0dafa884ad52ffb243625717a3471643e087

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
git -C "${VIME_INSTALL_ROOT}/MegatronAdaptor" apply --check "${PATCH_DIR}/megatron-adaptor.patch"
git -C "${VIME_INSTALL_ROOT}/MegatronAdaptor" apply "${PATCH_DIR}/megatron-adaptor.patch"
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

Build and install the Sunabako arm64 wheel before installing Vime requirements.
Use Rust 1.95, a C build toolchain, `curl`, `git`, and `python3-venv`. The Dockerfile
provides a separate wheel-build stage; for installation inside the base image:

```bash
export SUNABAKO_BUILD_ROOT="$(mktemp -d /tmp/vime-sunabako.XXXXXX)"
python3 -m venv "${SUNABAKO_BUILD_ROOT}/venv"
"${SUNABAKO_BUILD_ROOT}/venv/bin/python" -m pip install 'maturin>=1.8,<2'
curl -fsSL https://files.pythonhosted.org/packages/cd/16/d9fd84bbf784c19425d9d83d8d98deef6d5f107012e8d50bddeb5c11cf5c/sunabako-0.1.1.tar.gz \
  -o "${SUNABAKO_BUILD_ROOT}/sunabako.tar.gz"
mkdir "${SUNABAKO_BUILD_ROOT}/src"
tar -xf "${SUNABAKO_BUILD_ROOT}/sunabako.tar.gz" -C "${SUNABAKO_BUILD_ROOT}/src" --strip-components=1
git -C "${SUNABAKO_BUILD_ROOT}/src" apply --check "${VIME_INSTALL_ROOT}/vime/docker/patch/latest/sunabako-arm64.patch"
git -C "${SUNABAKO_BUILD_ROOT}/src" apply "${VIME_INSTALL_ROOT}/vime/docker/patch/latest/sunabako-arm64.patch"
(cd "${SUNABAKO_BUILD_ROOT}/src" && "${SUNABAKO_BUILD_ROOT}/venv/bin/maturin" build \
  --locked --release --target aarch64-unknown-linux-gnu --out "${SUNABAKO_BUILD_ROOT}/wheels")
python3 -m pip install --no-deps "${SUNABAKO_BUILD_ROOT}"/wheels/*.whl
```

Then install Vime and the training FLA packages:

```bash
pip install -r "${VIME_INSTALL_ROOT}/vime/requirements.txt"
pip install --no-deps --no-build-isolation -e "${VIME_INSTALL_ROOT}/vime"
pip install --no-deps einops 'fla-core==0.5.2' 'flash-linear-attention==0.5.2'
```

Training FLA 0.5.2 does not replace the serving FLA package required by vLLM-Ascend;
keep the latter as provided by the base image. Do not install the training package's
`npu` extra, which pins a different torch / torch-npu pair.

### 6. torch_memory_saver

Apply the NPU patch, then build and reinstall the wheel:

```bash
git clone --depth 1 --branch 2026.6.0 https://github.com/sgl-project/sgl-kernel-npu.git "${VIME_INSTALL_ROOT}/sgl-kernel-npu"
git -C "${VIME_INSTALL_ROOT}/sgl-kernel-npu" apply --check "${PATCH_DIR}/torch-memory-saver.patch"
git -C "${VIME_INSTALL_ROOT}/sgl-kernel-npu" apply "${PATCH_DIR}/torch-memory-saver.patch"
cd "${VIME_INSTALL_ROOT}/sgl-kernel-npu/contrib/torch_memory_saver/python"
python3 setup.py bdist_wheel
python3 -m pip install --no-deps --force-reinstall dist/torch_memory_saver-*.whl
cd "${VIME_INSTALL_ROOT}/vime"
```

### 7. vLLM Router

Build the pinned router with the shared health-check patch so Vime can own worker
health checks. This patch changes Rust as well as Python; applying only the Python
changes is insufficient. Install Rust 1.95, `cmake`, `pkg-config`, and `libssl-dev`
before building. The Dockerfile supplies these in a separate wheel-build stage.

```bash
git init "${VIME_INSTALL_ROOT}/vllm-router"
git -C "${VIME_INSTALL_ROOT}/vllm-router" remote add origin https://github.com/vllm-project/router.git
git -C "${VIME_INSTALL_ROOT}/vllm-router" fetch --depth 1 origin 1fbcde7443d75b36befb61bc081f64c2a1f13a4b
git -C "${VIME_INSTALL_ROOT}/vllm-router" checkout --detach FETCH_HEAD
git -C "${VIME_INSTALL_ROOT}/vllm-router" apply --check "${VIME_INSTALL_ROOT}/vime/docker/patch/latest/router-health-check.patch"
git -C "${VIME_INSTALL_ROOT}/vllm-router" apply "${VIME_INSTALL_ROOT}/vime/docker/patch/latest/router-health-check.patch"
pip install setuptools-rust wheel
cd "${VIME_INSTALL_ROOT}/vllm-router"
CARGO_BUILD_JOBS=4 python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python3 -m pip install --no-deps --force-reinstall dist/vllm_router-*.whl
cd "${VIME_INSTALL_ROOT}/vime"
```

Rebuild and reinstall the wheel when changing this patch; the source-only NPU
patch reconciliation does not rebuild Rust extensions.

## Environment Setup and Installation Check

Set the source paths before running Vime:

```bash
export PYTHONPATH="${VIME_INSTALL_ROOT}/Megatron-LM:${VIME_INSTALL_ROOT}/MegatronAdaptor:${VIME_INSTALL_ROOT}/TransformerEngineNPU:${VIME_INSTALL_ROOT}/vime${PYTHONPATH:+:${PYTHONPATH}}"

python3 -c 'import numpy, pyarrow, datasets, fastapi, starlette; from ray._private.serialization import SerializationContext; from ray._private.worker import global_worker; SerializationContext(global_worker); print(f"Data/Ray check passed: numpy={numpy.__version__}, pyarrow={pyarrow.__version__}, datasets={datasets.__version__}, fastapi={fastapi.__version__}, starlette={starlette.__version__}")'
python3 -c 'import megatron, mindspeed, megatron_adaptor, transformer_engine, torch_memory_saver, vime, vllm, vllm_ascend'
```

For a complete container build recipe, see [Dockerfile.npu](../Dockerfile.npu).
