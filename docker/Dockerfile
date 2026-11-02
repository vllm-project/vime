ARG BASE_IMAGE=vllm/vllm-openai@sha256:cb12c3016938a4ee98ed176ac833984089460689a0982cff4b5a60bb45b3cded
FROM rust:1.95-bookworm AS sunabako-wheel
ARG TARGETARCH
ARG PATCH_VERSION=latest
COPY docker/patch/${PATCH_VERSION}/sunabako-arm64.patch /tmp/sunabako-arm64.patch
RUN mkdir /wheels && if [ "${TARGETARCH}" = "arm64" ]; then \
    apt-get update && apt-get install -y --no-install-recommends python3-venv && \
    python3 -m venv /opt/sdk-build && /opt/sdk-build/bin/python -m pip install "maturin>=1.8,<2" && \
    curl -fsSL https://files.pythonhosted.org/packages/32/04/17067354c37b23f7fe00e70c131c863f85acaac86874baa36e32426fb8b1/sunabako-0.1.2.tar.gz -o /tmp/sdk.tar.gz && \
    echo "c7d8f67f165ed570cb7a28dc44813a9a15e097ca15ba915abd7db7834cfbf1cf  /tmp/sdk.tar.gz" | sha256sum -c - && \
    mkdir /sdk && tar -xf /tmp/sdk.tar.gz -C /sdk --strip-components=1 && cd /sdk && \
    git apply /tmp/sunabako-arm64.patch && \
    /opt/sdk-build/bin/maturin build --locked --release --target aarch64-unknown-linux-gnu --out /wheels; fi

FROM ${BASE_IMAGE} AS vllm-router-wheel
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo CARGO_BUILD_JOBS=4
COPY --from=rust:1.95-bookworm /usr/local/rustup /usr/local/rustup
COPY --from=rust:1.95-bookworm /usr/local/cargo/bin /usr/local/cargo/bin
ENV PATH=/usr/local/cargo/bin:${PATH}
ARG PATCH_VERSION=latest
ARG VLLM_ROUTER_COMMIT=1fbcde7443d75b36befb61bc081f64c2a1f13a4b
COPY docker/patch/${PATCH_VERSION}/router-health-check.patch /tmp/router-health-check.patch
COPY docker/patch/${PATCH_VERSION}/router-pd-registration.patch /tmp/router-pd-registration.patch
COPY docker/patch/${PATCH_VERSION}/router-worker-registration.patch /tmp/router-worker-registration.patch
RUN apt-get update && apt-get install -y --no-install-recommends git cmake pkg-config libssl-dev && \
    uv pip install --system setuptools-rust wheel && \
    git init /router && cd /router && git remote add origin https://github.com/vllm-project/router.git && \
    git fetch --depth 1 origin ${VLLM_ROUTER_COMMIT} && git checkout FETCH_HEAD && \
    git apply /tmp/router-health-check.patch && \
    git apply /tmp/router-pd-registration.patch && \
    git apply /tmp/router-worker-registration.patch && \
    uv build --python /usr/bin/python3 --wheel --no-build-isolation --out-dir /wheels

FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG VLLM_COMMIT=83c078d955cd6e0c63d3b12abe669c8de936a370
ARG MEGATRON_COMMIT=4b4acac9a1d28ea6829c8d4f566d75698a21249d
ARG DEEPGEMM_COMMIT=8965e4a5cded4041c8dbbe8d35b0be31fd8da90b
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST=9.0;10.0;10.3
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_VERSION=2.18.0
ARG TRANSFORMER_ENGINE_COMMIT=27486e03cfc1fa41f6932dcecdc47c71c47eac3e
ARG TRANSFORMER_ENGINE_CUDA_ARCHS=90;100a;103a
ARG TMS_COMMIT=bdab8c55f490a00a746ec34e301391617f5ca878
ARG TMS_CUDA_MAJOR=
ARG NCCL_VERSION=2.30.7
ARG MOONCAKE_VERSION=0.3.12
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

RUN mkdir -p /opt/vime && python -c 'import importlib.metadata as m; packages = ("vllm", "torch", "torchvision", "torchaudio", "cuda-python", "cuda-bindings", "flashinfer-python", "flashinfer-cubin", "transformers"); open("/opt/vime/constraints.txt", "w").write("".join(f"{p}=={m.version(p)}\n" for p in packages) + "pyarrow==25.0.1\n")'
ENV PIP_CONSTRAINT=/opt/vime/constraints.txt

# The training context-parallel stack uses FA2 + FA3.
COPY docker/patch/${PATCH_VERSION}/flash-attn-cxx20.patch /tmp/flash-attn-cxx20.patch
RUN curl -fsSL https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz -o /tmp/flash_attn-2.8.3.tar.gz && \
    echo "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d  /tmp/flash_attn-2.8.3.tar.gz" | sha256sum -c - && \
    tar -xf /tmp/flash_attn-2.8.3.tar.gz -C /tmp && \
    cd /tmp/flash_attn-2.8.3 && git apply /tmp/flash-attn-cxx20.patch && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=${FA2_MAX_JOBS} pip -v install . --no-build-isolation --no-deps && \
    cd /root && rm -rf /tmp/flash_attn-2.8.3 /tmp/flash_attn-2.8.3.tar.gz /tmp/flash-attn-cxx20.patch

# This FA3 commit provides the window_size_left/window_size_right API used by TE.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
    cd hopper/ && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
    cd /root/ && rm -rf flash-attention/

RUN pip install flash-linear-attention==0.5.2 tilelang==0.1.12 scikit-build-core
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

RUN apt-get update && apt-get install -y libdw-dev libelf-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/zhuzilin/DeepGEMM.git --recursive && \
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
    pip install -r /tmp/requirements.txt "wandb==0.31.0" "opentelemetry-exporter-otlp==1.45.1"
RUN pip install --no-deps --force-reinstall "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
COPY --from=vllm-router-wheel /wheels/ /tmp/router-wheels/
RUN pip install --no-deps --force-reinstall /tmp/router-wheels/*.whl && rm -rf /tmp/router-wheels

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
COPY docker/patch/${PATCH_VERSION}/vllm-score-centering.patch /tmp/vllm-score-centering.patch
COPY docker/patch/${PATCH_VERSION}/vllm-fine-grained-tracing.patch /tmp/vllm-fine-grained-tracing.patch
COPY docker/patch/${PATCH_VERSION}/vllm-generation-health.patch /tmp/vllm-generation-health.patch
COPY docker/patch/${PATCH_VERSION}/vllm-tensor-dump.patch /tmp/vllm-tensor-dump.patch
COPY docker/patch/${PATCH_VERSION}/vllm-deepgemm-bf16.patch /tmp/vllm-deepgemm-bf16.patch
COPY docker/patch/${PATCH_VERSION}/vllm-deepep-optional-shrink.patch /tmp/vllm-deepep-optional-shrink.patch
COPY docker/patch/${PATCH_VERSION}/vllm-glm5next-unbounded-gate.patch /tmp/vllm-glm5next-unbounded-gate.patch
COPY docker/patch/${PATCH_VERSION}/vllm-tito-columnar-logprobs-render.patch /tmp/vllm-tito-columnar-logprobs-render.patch
COPY docker/patch/${PATCH_VERSION}/vllm-watchdog-fatal-action.patch /tmp/vllm-watchdog-fatal-action.patch
RUN VLLM_SITE="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" && \
    cd "$VLLM_SITE" && \
    git apply -v /tmp/vllm-pull_weights.patch && \
    git apply -v /tmp/vllm-score-centering.patch && \
    git apply -v /tmp/vllm-fine-grained-tracing.patch && \
    git apply -v /tmp/vllm-generation-health.patch && \
    git apply -v /tmp/vllm-tensor-dump.patch && \
    git apply -v /tmp/vllm-deepgemm-bf16.patch && \
    git apply -v /tmp/vllm-deepep-optional-shrink.patch && \
    git apply -v /tmp/vllm-glm5next-unbounded-gate.patch && \
    git apply -v /tmp/vllm-tito-columnar-logprobs-render.patch && \
    git apply -v /tmp/vllm-watchdog-fatal-action.patch && \
    rm /tmp/vllm-pull_weights.patch \
       /tmp/vllm-score-centering.patch \
       /tmp/vllm-fine-grained-tracing.patch \
       /tmp/vllm-generation-health.patch \
       /tmp/vllm-tensor-dump.patch \
       /tmp/vllm-deepgemm-bf16.patch \
       /tmp/vllm-deepep-optional-shrink.patch \
       /tmp/vllm-glm5next-unbounded-gate.patch \
       /tmp/vllm-tito-columnar-logprobs-render.patch \
       /tmp/vllm-watchdog-fatal-action.patch

# ====================================== Install main package ============================================

ARG VIME_REPO=https://github.com/vllm-project/vime.git
ARG VIME_COMMIT=main
RUN git clone ${VIME_REPO} /root/vime && \
    cd /root/vime && \
    git checkout ${VIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/vime/vime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation --no-deps

COPY docker/cuda-compat.sh docker/entrypoint.sh /opt/vime/
ARG CUDA_COMPAT_VERSION=580.178.04-1ubuntu1
RUN apt-get update && apt-get install -y --no-install-recommends "cuda-compat-13-0=${CUDA_COMPAT_VERSION}" && rm -rf /var/lib/apt/lists/*
RUN chmod +x /opt/vime/entrypoint.sh
ENTRYPOINT ["/opt/vime/entrypoint.sh"]

RUN CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    pip install --no-deps --force-reinstall "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}"
CMD ["/bin/bash"]
