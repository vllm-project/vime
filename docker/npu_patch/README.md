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

### Python Version

Only `python==3.11` is supported currently.

```shell
conda create -n vime_release python=3.11
conda activate vime_release
```

### Working Directory Setup

```shell
mkdir <WORKSPACE> && cd <WORKSPACE>
```

### CANN Environment

Prior to start work with vime on Ascend you need to install CANN Toolkit, Kernels operator package and NNAL version 9.0.0 , check the [installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns\&InstallType=local\&OS=openEuler\&Software=cannToolKit)

```shell
source <CANN_PATH>/ascend-toolkit/set_env.sh
source <CANN_PATH>/nnal/atb/set_env.sh
```

### PyTorch and PyTorch NPU

```shell
pip install torch-npu==2.10.0
```

## Installing Dependencies


### Megatron-Bridge

```shell
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

cd <WORKSPACE>
git clone https://github.com/radixark/Megatron-Bridge.git -b bridge
git checkout 3fd3768045422d0aa5c97e90a4e6c659aea9acb9
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
```

### Megatron-LM

```shell
cd <WORKSPACE>
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
  cd Megatron-LM/ && git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862 && \
  pip install -e .
```

### MindSpeed

```shell
cd <WORKSPACE>
git clone https://gitcode.com/Ascend/MindSpeed.git && \
  cd MindSpeed/ && git checkout fc63de5c48426dd019c3b3f39e65f5bdf56e4086 && \
  pip install -e .
```

### Vime

```shell
cd <WORKSPACE>
git clone https://github.com/vllm-project/vime.git -b wxx-ascend && cd vime
cp -r docker/npu_patch ../npu_patch
pip install -e .
```

## Applying Patches

```shell
cd <WORKSPACE>/Megatron-LM
git apply ../npu_patch/megatron_common.patch
git apply ../npu_patch/megatron.patch

cd <WORKSPACE>/Megatron-Bridge
git apply ../npu_patch/megatron-bridge.patch

cd <WORKSPACE>/MindSpeed
git apply ../npu_patch/mindspeed.patch
```

## Additional Dependencies

```shell
cd <WORKSPACE>/vime
pip install triton-ascend
pip install torch-npu==2.10.0
pip install torchvision==0.25.0
pip install numpy==1.26.4
```

