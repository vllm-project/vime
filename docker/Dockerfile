ARG BASE_IMAGE=vllm/vllm-openai:nightly
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG DEEPGEMM_COMMIT=b38a77cd193cf38f670caae192310521d24343be
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST=9.0;10.0;10.3
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_COMMIT=c9877beb87ad7e711e1869dd0b5062167ede447a
ARG TRANSFORMER_ENGINE_CUDA_ARCHS=90;100a;103a
ARG TMS_COMMIT=8d30c59ca12a68d9deccbc9c6599076a1218cbc5

ARG ENABLE_CUDA_13=1
ARG FA2_MAX_JOBS=64

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

# vllm/vllm-openai base is an inference image — add CUDA dev headers + cmake/git
# so TE / apex / flash-attn source builds find cusparse.h etc.
# Dev packages must match the base toolkit: -12-9 for cu129, -13-0 for cu130.
RUN apt-get update && apt-get install -y \
    nvtop rsync dnsutils prometheus git cmake && \
    if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      apt-get install -y \
        cuda-nvrtc-dev-13-0 cuda-nvml-dev-13-0 cuda-profiler-api-13-0 cuda-nvtx-13-0 \
        libcusparse-dev-13-0 libcusolver-dev-13-0 libcufft-dev-13-0 libcurand-dev-13-0 \
        libcudnn9-dev-cuda-13; \
    else \
      apt-get install -y \
        cuda-nvrtc-dev-12-9 cuda-nvml-dev-12-9 cuda-profiler-api-12-9 cuda-nvtx-12-9 \
        libcusparse-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9 \
        libcudnn9-dev-cuda-12; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# vllm/vllm-openai base only ships python3; subsequent source builds invoke `python`.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ====================================== Python dependencies ============================================

# The validated TransformerEngine 2.16 context-parallel stack uses FA2 + FA3.
RUN pip uninstall -y flash-attn-4 flash_attn_4 || true
RUN MAX_JOBS=${FA2_MAX_JOBS} pip -v install flash-attn==2.8.3 --no-build-isolation

# This FA3 commit provides the window_size_left/window_size_right API used by TE 2.16.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
    cd hopper/ && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
    cd /root/ && rm -rf flash-attention/

RUN pip install flash-linear-attention==0.4.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
ARG INSTALL_FLASHQLA=0
RUN if [ "${INSTALL_FLASHQLA}" = "1" ]; then \
      pip install git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} --no-build-isolation; \
    else \
      echo "Skipping FlashQLA (INSTALL_FLASHQLA=0; sm90/Hopper-only — use --qwen-gdn-backend fla)"; \
    fi
# cublas dev header for TE CMake (arm64 base ships runtime .so but not the header).
# cu13 also needs the -13-0 headers that TE's nvcc build expects.
RUN apt-get update && \
    if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      apt-get install -y libcublas-dev-13-0; \
    else \
      apt-get install -y libcublas-dev-12-9; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# The cu13 TE wheel is built against a newer cuBLAS than the vLLM CUDA 13.0
# base. Build the pinned source revision against this image.
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install pybind11 && \
      NCCL_INCLUDE_DIR="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.nccl").submodule_search_locations)) + "/include")')" && \
      export CPATH="${NCCL_INCLUDE_DIR}${CPATH:+:${CPATH}}" && \
      NVTE_CUDA_ARCHS="${TRANSFORMER_ENGINE_CUDA_ARCHS}" MAX_JOBS=64 \
        pip -v install --no-build-isolation \
        git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_COMMIT}; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.16.1"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT}

# zhuzilin fork builds, grouped together right after Megatron-LM:
# torch_memory_saver, plus the GLM-5 train/rollout alignment kernels.
RUN TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} --no-cache-dir --force-reinstall

RUN git clone https://github.com/zhuzilin/DeepGEMM.git --recursive && \
    cd DeepGEMM && git checkout ${DEEPGEMM_COMMIT} && \
    git submodule update --init --recursive && \
    bash build_sgl_deep_gemm.sh && \
    pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl && \
    cd /root/ && rm -rf DeepGEMM

RUN git clone https://github.com/zhuzilin/DeepEP.git /root/DeepEP && \
    cd /root/DeepEP && git checkout ${DEEPEP_COMMIT} && \
    CUDA_ARCHS="${DEEPEP_CUDA_ARCH_LIST}" && \
    if [ -d /usr/local/cuda/include/cccl ]; then \
      export CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}"; \
      export NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include/cccl ${NVCC_PREPEND_FLAGS:-}"; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" MAX_JOBS=64 python setup.py bdist_wheel && \
    pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
    cd /root/ && rm -rf DeepEP
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# https://github.com/pytorch/pytorch/issues/168167
# cu130 base already ships nvidia-cudnn-cu13 9.19.0; only cu129 needs the pin.
RUN if [ "${ENABLE_CUDA_13}" != "1" ]; then \
      pip install nvidia-cudnn-cu12==9.16.0.29; \
    fi

# Megatron requires NumPy 1.x, while SciPy 1.18+ requires NumPy 2.x.
RUN pip install "numpy==1.26.4" "scipy==1.17.1"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron*.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm -f megatron*.patch && \
    pip install -e .

# Patch vLLM with vime's local fixes. vLLM is a pip install (not a git checkout)
# so apply with plain `git apply` (no --3way). Pull-weights lands first because
# the general patch also updates gpu_worker.py against the resulting line layout.
COPY docker/patch/${PATCH_VERSION}/vllm-pull_weights.patch /tmp/vllm-pull_weights.patch
COPY docker/patch/${PATCH_VERSION}/vllm.patch /tmp/vllm.patch
RUN VLLM_SITE="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" && \
    cd "$VLLM_SITE" && \
    git apply -v /tmp/vllm-pull_weights.patch && \
    git apply -v --allow-empty /tmp/vllm.patch && \
    rm /tmp/vllm-pull_weights.patch /tmp/vllm.patch

# ====================================== Install main package ============================================

ARG VIME_REPO=https://github.com/vllm-project/vime.git
ARG VIME_COMMIT=main
RUN git clone ${VIME_REPO} /root/vime && \
    cd /root/vime && \
    git checkout ${VIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/vime/vime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

# Reset ENTRYPOINT inherited from the vllm/vllm-openai base (`vllm serve`), so the
# image is a plain bash/ray environment. Without this, `docker run ... bash -c ...`
# and `ray job submit` append to `vllm serve` and break.
ENTRYPOINT []
CMD ["/bin/bash"]
