# Vime NPU Patch Installation Guide

This guide provides instructions for installing Vime with NPU support, including all required dependencies and patches.

> S7 closeout (2026-09-09): retain Ascend #385 (training stack) and #396
> (torch_dist/ref-load), with native HF loading and main's shared orchestration.
> Revert #409 (`f5b84916`) and its follow-up Qwen3.5 NPU adaptations; defer that
> model to the next stage in a fresh, matched environment. Main's Qwen3.5 model
> code is retained. Existing Qwen3-4B, Qwen3-30B-A3B, Qwen3-VL-8B and the 30B
> torch_dist/ref-load run passed before the Qwen3.5 environment changes; this
> does not certify a fresh image or a post-revert E2E run. No installed packages
> or vendor source trees are rolled back as part of this source-only closeout.
> Post-revert checks: 183 grouped CPU tests passed. Common → NPU Megatron
> patches and the reverted Bridge patch pass apply checks on their pinned
> clean source revisions. Serving patches and `docker/patch/latest` are unchanged.

## Component Version Mapping

| Component       | Version/Commit                           | Source                                                                                                              |
| --------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| vime            | main                                     | [GitHub](https://github.com/vllm-project/vime/tree/main)                                                            |
| vLLM | e6bfe03ad73a3330cb427885aa90d97a12e1c704 + NPU patch | S6 serving baseline, retained for S7 |
| vLLM-Ascend | fd815467c221ee600137f6bdd53fe354d5e7c999 + NPU patch | S6 serving baseline, retained for S7 |
| Megatron-Bridge | 3fd3768045422d0aa5c97e90a4e6c659aea9acb9 | [GitHub](https://github.com/radixark/Megatron-Bridge)                                                               |
| Megatron-LM     | 1dcf0dafa884ad52ffb243625717a3471643e087 | [GitHub](https://github.com/NVIDIA/Megatron-LM)                                                                     |
| MegatronAdaptor | 15582addff3f3d4680e350826fa70d012b475509 | [GitCode](https://gitcode.com/Ascend/MegatronAdaptor)                                                               |
| TransformerEngineNPU | d743c83d060d5edc48867ecb9e93ec80d81860e4 | [GitCode](https://gitcode.com/Ascend/TransformerEngineNPU)                                                          |
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

Vime's Ascend NPU adaptation lives on the **`ascend`** branch, so clone that
branch (not `main`):

```bash
git clone --branch ascend https://github.com/vllm-project/vime.git "${WORKSPACE}/vime"
export PATCH_DIR="${WORKSPACE}/vime/docker/npu_patch"
```

#### 1. Megatron-Bridge (legacy build dependency, not the native loader)

The source PR used this via `PYTHONPATH` (no editable install) and required
`nvidia-modelopt`. This is not a prerequisite for Vime's native HF loader;
whether to retain it in the S7 image remains under review.

```bash
export MEGATRON_BRIDGE_COMMIT=3fd3768045422d0aa5c97e90a4e6c659aea9acb9
export MBRIDGE_COMMIT=89eb10887887bc74853f89a4de258c0702932a1c
pip install "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps
git clone --branch bridge https://github.com/radixark/Megatron-Bridge.git "${WORKSPACE}/Megatron-Bridge"
git -C "${WORKSPACE}/Megatron-Bridge" checkout "${MEGATRON_BRIDGE_COMMIT}"

git -C "${WORKSPACE}/Megatron-Bridge" apply --whitespace=nowarn "${PATCH_DIR}/megatron-bridge.patch"

pip install --no-build-isolation "nvidia-modelopt[torch]>=0.37.0"
```

#### 2. Megatron-LM

```bash
export MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
git clone https://github.com/NVIDIA/Megatron-LM.git "${WORKSPACE}/Megatron-LM"
git -C "${WORKSPACE}/Megatron-LM" checkout "${MEGATRON_COMMIT}"

git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${WORKSPACE}/vime/docker/patch/latest/megatron.patch"
git -C "${WORKSPACE}/Megatron-LM" apply --whitespace=nowarn "${PATCH_DIR}/megatron.patch"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/Megatron-LM"
```

#### 3. MegatronAdaptor and TransformerEngineNPU

The NPU training stack now uses the two source repositories directly. The mainline Megatron patch is applied first; `docker/npu_patch/megatron.patch` contains only the NPU-specific changes rebased onto that mainline patch:

pip install --no-deps --no-build-isolation -e ${WORKSPACE}/MegatronAdaptor
pip install --no-deps --no-build-isolation -e ${WORKSPACE}/TransformerEngineNPU

Do not install the CUDA TransformerEngine package in the same environment.

#### 4. MindSpeed

```bash
export MINDSPEED_COMMIT=fc63de5c48426dd019c3b3f39e65f5bdf56e4086
git clone https://gitcode.com/Ascend/MindSpeed.git "${WORKSPACE}/MindSpeed"
git -C "${WORKSPACE}/MindSpeed" checkout "${MINDSPEED_COMMIT}"

git -C "${WORKSPACE}/MindSpeed" apply --whitespace=nowarn "${PATCH_DIR}/mindspeed.patch"

pip install --no-deps --no-build-isolation -e "${WORKSPACE}/MindSpeed"
```

#### 5. Vime

```bash
pip install -r "${WORKSPACE}/vime/requirements.txt"
pip install "vllm-router>=0.1.14"
pip install --no-deps --no-build-isolation -e "${WORKSPACE}/vime"
```

The NPU training region and optimizer state use Ascend `torch_memory_saver`.
Retain the working build in an existing environment; the source build recipe is:

```bash
git clone --branch 2026.6.0 https://github.com/sgl-project/sgl-kernel-npu.git "${WORKSPACE}/sgl-kernel-npu"
cd "${WORKSPACE}/sgl-kernel-npu"
bash build.sh -a kernels
bash build.sh -a memory-saver
pip install --no-deps output/torch_memory_saver-0.0.8-cp312-cp312-linux_aarch64.whl
```

#### 5. Install vLLM and vLLM Ascend

```bash
export VLLM_COMMIT=e6bfe03ad73a3330cb427885aa90d97a12e1c704
export VLLM_ASCEND_COMMIT=fd815467c221ee600137f6bdd53fe354d5e7c999

git clone https://github.com/vllm-project/vllm.git "${WORKSPACE}/vllm"
git -C "${WORKSPACE}/vllm" checkout "${VLLM_COMMIT}"
VLLM_TARGET_DEVICE=empty pip install -v -e "${WORKSPACE}/vllm"

git clone https://github.com/vllm-project/vllm-ascend.git "${WORKSPACE}/vllm-ascend"
git -C "${WORKSPACE}/vllm-ascend" checkout "${VLLM_ASCEND_COMMIT}"
git -C "${WORKSPACE}/vllm-ascend" submodule update --init --recursive
pip install -v -e "${WORKSPACE}/vllm-ascend"
```

Apply `vllm.patch` and `vllm-ascend.patch` to those exact revisions before
validation. Do not replace the existing source trees during conflict resolution.

For image patch reconciliation, persist the common Megatron patch as
`/opt/npu_patch/megatron-common.patch`. `series.conf` applies common → NPU and
reverts in reverse order. This reconciles patch bytes, not repository versions;
an old Megatron checkout cannot be upgraded by the patch reconciler alone.

## Additional Dependencies

The source PR specified the following versions. They are not an instruction to
upgrade the existing S6 environment; in particular, validate the new NPU kernel
requirements before changing torch-npu:

```shell
pip install torch-npu==2.10.0
pip install torchvision==0.25.0
pip install numpy==1.26.4
```
