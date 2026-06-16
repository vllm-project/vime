# Vime NPU Patch Installation Guide

This guide provides instructions for installing Vime with NPU support, including all required dependencies and patches.

## Component Version Mapping

| Component       | Version/Commit                           | Source                                                                                                              |
| --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| vime            | main                                     | [GitHub](https://github.com/vllm-project/vime/tree/main)                                                            |
| Megatron-Bridge | 3fd3768045422d0aa5c97e90a4e6c659aea9acb9 | [GitHub](https://github.com/radixark/Megatron-Bridge)                                                               |
| Megatron-LM     | 3714d81d418c9f1bca4594fc35f9e8289f652862 | [GitHub](https://github.com/NVIDIA/Megatron-LM)                                                                     |
| MindSpeed       | fc63de5c48426dd019c3b3f39e65f5bdf56e4086 | [GitCode](https://gitcode.com/Ascend/MindSpeed)                                                                     |
| HDK             | 25.3.RC1                                 | [Ascend](https://www.hiascend.com/hardware/firmware-drivers/commercial?product=7\&model=33)                         |
| CANN            | 9.0.0                                    | [Ascend](https://www.hiascend.com/developer/download/community/result?module=cann\&cann=9.0.0\&product=7\&model=33) |

## Preparing the Running Environment

Run the steps below in a Python 3.12 environment with CANN 9.0.0. A
`quay.io/ascend/vllm-ascend:nightly-main-a3` container can be used as the base.

```bash
export WORKSPACE=/root
cd "${WORKSPACE}"
```

Vime's Ascend NPU adaptation lives on the **`npu`** branch, so clone that
branch (not `main`):

```bash
git clone --branch npu https://github.com/vllm-project/vime.git "${WORKSPACE}/vime"
export PATCH_DIR="${WORKSPACE}/vime/docker/npu_patch"
```


#### 1. Megatron-Bridge

Used via `PYTHONPATH` (no editable install); it requires `nvidia-modelopt`.

```bash
export MEGATRON_BRIDGE_COMMIT=3fd3768045422d0aa5c97e90a4e6c659aea9acb9
export MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
pip install "git+https://ghproxy.com/https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps
git clone --branch bridge https://github.com/radixark/Megatron-Bridge.git "${WORKSPACE}/Megatron-Bridge"
git -C "${WORKSPACE}/Megatron-Bridge" checkout "${MEGATRON_BRIDGE_COMMIT}"

git -C "${WORKSPACE}/Megatron-Bridge" apply --whitespace=nowarn "${PATCH_DIR}/megatron-bridge.patch"

pip install --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
```


#### 2. Megatron-LM

```bash
export MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
git clone https://github.com/NVIDIA/Megatron-LM.git "${WORKSPACE}/Megatron-LM"
git -C "${WORKSPACE}/Megatron-LM" checkout "${MEGATRON_COMMIT}"

git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/meagtron_comm.patch"
git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/megatron.patch"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/Megatron-LM"
```

#### 3. MindSpeed

```bash
export MINDSPEED_COMMIT=fc63de5c48426dd019c3b3f39e65f5bdf56e4086
git clone https://gitcode.com/Ascend/MindSpeed.git "${WORKSPACE}/MindSpeed"
git -C "${WORKSPACE}/MindSpeed" checkout "${MINDSPEED_COMMIT}"

git -C "${WORKSPACE}/MindSpeed" apply --whitespace=nowarn "${PATCH_DIR}/mindspeed.patch"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/MindSpeed"
```


#### 4. Vime

```bash
pip install -r "${WORKSPACE}/vime/requirements.txt"
pip install "vllm-router>=0.1.14"
pip install --no-deps --no-build-isolation -e "${WORKSPACE}/vime"
```

Build the matching Ascend `torch_memory_saver` wheel. NPU does not actually use
`torch_memory_saver`, but the code still imports and calls it and will break
without it, and there is currently no published Python 3.12 build — so compile
it from source:

```bash
git clone --branch 2026.6.0 https://github.com/sgl-project/sgl-kernel-npu.git "${WORKSPACE}/sgl-kernel-npu"
cd "${WORKSPACE}/sgl-kernel-npu"
bash build.sh -a kernels
bash build.sh -a memory-saver
pip install --no-deps output/torch_memory_saver-0.0.8-cp312-cp312-linux_aarch64.whl
```

#### 5. Install vLLM and vLLM Ascend


```bash
export VLLM_COMMIT=9090368b650896bf5fc990c921df7eb4c20355a5

git clone https://github.com/vllm-project/vllm.git "${WORKSPACE}/vllm"
git -C "${WORKSPACE}/vllm" checkout "${VLLM_COMMIT}"
VLLM_TARGET_DEVICE=empty pip install -v -e "${WORKSPACE}/vllm"

git clone https://github.com/vllm-project/vllm-ascend.git "${WORKSPACE}/vllm-ascend"
git -C "${WORKSPACE}/vllm-ascend" submodule update --init --recursive
pip install -v -e "${WORKSPACE}/vllm-ascend"
```

> [!NOTE]
> vLLM Ascend has not yet cut a release tag against vLLM 0.22.0. As a temporary
> measure we pin vLLM to the commit below and build vLLM Ascend from source.
> Once vLLM Ascend officially supports 0.22.0, this whole step can be omitted and
> the released packages used instead.

## Additional Dependencies

Ensure the following packages are pinned to these matching versions：

```shell
pip install torch-npu==2.10.0
pip install torchvision==0.25.0
pip install numpy==1.26.4
```

