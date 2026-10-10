ARG BASE_IMAGE=vllm/vllm-openai@sha256:f8696a5176c1fb61e025b0772505674a3e092c25597eb26117003945b5121e8d
FROM rust:1.95-bookworm AS vllm-rust-frontend
ENV CARGO_BUILD_JOBS=4
ARG PATCH_VERSION=latest
ARG VLLM_COMMIT=43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5
ARG VLLM_RS_BUILD_VERSION=0.31.1rc1.dev23+g43b4aaea3
COPY docker/patch/${PATCH_VERSION}/vllm-rust-frontend.patch /tmp/vllm-rust-frontend.patch
COPY docker/patch/${PATCH_VERSION}/vllm-rust-observability.patch /tmp/vllm-rust-observability.patch
RUN git init /build && cd /build && \
    git remote add origin https://github.com/vllm-project/vllm.git && \
    git fetch --depth 1 origin ${VLLM_COMMIT} && git checkout FETCH_HEAD && \
    git apply /tmp/vllm-rust-frontend.patch && \
    git apply /tmp/vllm-rust-observability.patch
RUN cd /build && VLLM_RS_BUILD_VERSION=${VLLM_RS_BUILD_VERSION} \
    cargo build --locked --release --manifest-path rust/Cargo.toml \
    -p vllm-cmd --bin vllm-rs --features native-tls-vendored

FROM rust:1.95-bookworm AS sunabako-wheel
ARG TARGETARCH
ARG PATCH_VERSION=latest
COPY docker/patch/${PATCH_VERSION}/sunabako-arm64.patch /tmp/sunabako-arm64.patch
RUN mkdir /wheels && if [ "${TARGETARCH}" = "arm64" ]; then \
    apt-get update && apt-get install -y --no-install-recommends python3-venv && \
    python3 -m venv /opt/sdk-build && /opt/sdk-build/bin/python -m pip install "maturin>=1.8,<2" && \
    curl -fsSL https://files.pythonhosted.org/packages/cd/16/d9fd84bbf784c19425d9d83d8d98deef6d5f107012e8d50bddeb5c11cf5c/sunabako-0.1.1.tar.gz -o /tmp/sdk.tar.gz && \
    echo "b4057ea011e61ad942249e8156c54aad38c7f5f73d0075c93f3005f70d66036d  /tmp/sdk.tar.gz" | sha256sum -c - && \
    mkdir /sdk && tar -xf /tmp/sdk.tar.gz -C /sdk --strip-components=1 && cd /sdk && \
    git apply /tmp/sunabako-arm64.patch && \
    /opt/sdk-build/bin/maturin build --locked --release --target aarch64-unknown-linux-gnu --out /wheels; fi

FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG VLLM_COMMIT=43b4aaea3e40e20ef53ce2bfecfa1b72060bf2b5
ARG MEGATRON_COMMIT=4b4acac9a1d28ea6829c8d4f566d75698a21249d
ARG DEEPGEMM_COMMIT=b38a77cd193cf38f670caae192310521d24343be
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST=9.0;10.0;10.3
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_VERSION=2.18.0
ARG TRANSFORMER_ENGINE_COMMIT=27486e03cfc1fa41f6932dcecdc47c71c47eac3e
ARG TRANSFORMER_ENGINE_CUDA_ARCHS=90;100a;103a
ARG TMS_COMMIT=4d525cf378fdbfe7eb044909b8e66e112a508624
ARG TMS_CUDA_MAJOR=
ARG NCCL_VERSION=2.30.7
ARG MAX_JOBS=16
ENV MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

ARG ENABLE_CUDA_13=1
ARG FA2_MAX_JOBS=64

# ======================================== Setup =============================================

WORKDIR /root/
RUN python3 -c 'import os, vllm; assert os.environ["VLLM_COMMIT"][:9] in vllm.__version__, (os.environ["VLLM_COMMIT"], vllm.__version__)'

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
RUN uv pip install --system pip && ln -sf /usr/bin/python3 /usr/local/bin/python
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ====================================== Python dependencies ============================================

# The training context-parallel stack uses FA2 + FA3.
RUN pip uninstall -y flash-attn-4 flash_attn_4 || true
RUN MAX_JOBS=${FA2_MAX_JOBS} pip -v install flash-attn==2.8.3 --no-build-isolation

# This FA3 commit provides the window_size_left/window_size_right API used by TE.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
    cd hopper/ && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
    cd /root/ && rm -rf flash-attention/

RUN pip install flash-linear-attention==0.5.2
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
      NVTE_WITH_NCCL_EP=0 NVTE_CUDA_ARCHS="${TRANSFORMER_ENGINE_CUDA_ARCHS}" \
        pip -v install --no-build-isolation \
        git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_COMMIT}; \
    else \
      NVTE_WITH_NCCL_EP=0 pip -v install --no-build-isolation "transformer_engine[pytorch]==${TRANSFORMER_ENGINE_VERSION}"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 1" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT}

# zhuzilin fork builds, grouped together right after Megatron-LM:
# torch_memory_saver, plus the GLM-5 train/rollout alignment kernels.
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} --no-cache-dir --force-reinstall --no-build-isolation

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
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" python setup.py bdist_wheel && \
    pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
    cd /root/ && rm -rf DeepEP
RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
COPY --from=sunabako-wheel /wheels/ /tmp/sunabako-wheels/
RUN if ls /tmp/sunabako-wheels/*.whl >/dev/null 2>&1; then pip install --no-deps /tmp/sunabako-wheels/*.whl; fi
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# https://github.com/pytorch/pytorch/issues/168167
# cu130 base already ships nvidia-cudnn-cu13 9.19.0; only cu129 needs the pin.
RUN if [ "${ENABLE_CUDA_13}" != "1" ]; then \
      pip install nvidia-cudnn-cu12==9.16.0.29; \
    fi

RUN pip install "scipy==1.17.1" "kernels<0.15.0"

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

# Patch vLLM with vime's local fixes. vLLM is a pip install (not a git checkout),
# so apply the independently maintained patches in their validated order.
COPY docker/patch/${PATCH_VERSION}/vllm-pull_weights.patch /tmp/vllm-pull_weights.patch
COPY docker/patch/${PATCH_VERSION}/vllm.patch /tmp/vllm.patch
COPY docker/patch/${PATCH_VERSION}/vllm-score-centering.patch /tmp/vllm-score-centering.patch
COPY docker/patch/${PATCH_VERSION}/vllm-pd-request-metrics.patch /tmp/vllm-pd-request-metrics.patch
COPY docker/patch/${PATCH_VERSION}/vllm-inflight-queue-diagnostics.patch /tmp/vllm-inflight-queue-diagnostics.patch
COPY docker/patch/${PATCH_VERSION}/vllm-retained-serving.patch /tmp/vllm-retained-serving.patch
COPY docker/patch/${PATCH_VERSION}/vllm-generation-health.patch /tmp/vllm-generation-health.patch
COPY docker/patch/${PATCH_VERSION}/vllm-tensor-dump.patch /tmp/vllm-tensor-dump.patch
COPY docker/patch/${PATCH_VERSION}/vllm-fine-grained-tracing.patch /tmp/vllm-fine-grained-tracing.patch
COPY docker/patch/${PATCH_VERSION}/vllm-nixl-peer-retirement.patch /tmp/vllm-nixl-peer-retirement.patch
COPY docker/patch/${PATCH_VERSION}/vllm-mooncake-prom-metrics.patch /tmp/vllm-mooncake-prom-metrics.patch
COPY docker/patch/${PATCH_VERSION}/vllm-host-cache-ratio.patch /tmp/vllm-host-cache-ratio.patch
COPY docker/patch/${PATCH_VERSION}/vllm-deepgemm-bf16.patch /tmp/vllm-deepgemm-bf16.patch
COPY docker/patch/${PATCH_VERSION}/vllm-deepep-optional-shrink.patch /tmp/vllm-deepep-optional-shrink.patch
RUN VLLM_SITE="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" && \
    cd "$VLLM_SITE" && \
    git apply -v /tmp/vllm-pull_weights.patch && \
    git apply -v --allow-empty /tmp/vllm.patch && \
    git apply -v /tmp/vllm-score-centering.patch && \
    git apply -v /tmp/vllm-pd-request-metrics.patch && \
    git apply -v /tmp/vllm-inflight-queue-diagnostics.patch && \
    git apply -v /tmp/vllm-retained-serving.patch && \
    git apply -v /tmp/vllm-generation-health.patch && \
    git apply -v /tmp/vllm-tensor-dump.patch && \
    git apply -v /tmp/vllm-fine-grained-tracing.patch && \
    git apply -v /tmp/vllm-nixl-peer-retirement.patch && \
    git apply -v /tmp/vllm-mooncake-prom-metrics.patch && \
    git apply -v /tmp/vllm-host-cache-ratio.patch && \
    git apply -v /tmp/vllm-deepgemm-bf16.patch && \
    git apply -v /tmp/vllm-deepep-optional-shrink.patch && \
    rm /tmp/vllm-pull_weights.patch /tmp/vllm.patch \
       /tmp/vllm-score-centering.patch \
       /tmp/vllm-pd-request-metrics.patch \
       /tmp/vllm-inflight-queue-diagnostics.patch \
       /tmp/vllm-retained-serving.patch \
       /tmp/vllm-generation-health.patch \
       /tmp/vllm-tensor-dump.patch \
       /tmp/vllm-fine-grained-tracing.patch \
       /tmp/vllm-nixl-peer-retirement.patch \
       /tmp/vllm-mooncake-prom-metrics.patch \
       /tmp/vllm-host-cache-ratio.patch \
       /tmp/vllm-deepgemm-bf16.patch \
       /tmp/vllm-deepep-optional-shrink.patch

# ====================================== Install main package ============================================
COPY --from=vllm-rust-frontend /build/rust/target/release/vllm-rs /usr/local/lib/python3.12/dist-packages/vllm/vllm-rs

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

RUN CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    pip install --no-deps --force-reinstall "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}"
CMD ["/bin/bash"]
