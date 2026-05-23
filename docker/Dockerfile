ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0-cu129-ubuntu2404
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG SGLANG_VERSION=0.5.10.post1

ARG ENABLE_CUDA_13=0

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

# vllm/vllm-openai base is an inference image — add cu12 dev headers + cmake/git
# so TE / apex / flash-attn source builds find cusparse.h etc.
RUN apt-get update && apt-get install -y \
    nvtop rsync dnsutils git cmake \
    cuda-nvrtc-dev-12-9 cuda-nvml-dev-12-9 cuda-profiler-api-12-9 cuda-nvtx-12-9 \
    libcusparse-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9 \
    libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# vllm/vllm-openai base only ships python3; subsequent source builds invoke `python`.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ====================================== Python dependencies ============================================

# The compilation is slow, thus should be put at top
# TransformerEngines does not support too high FA2
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# The compilation is slow, thus should be put at top
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && git submodule update --init && cd hopper/ && \
    MAX_JOBS=96 python setup.py install && \
    export python_path=`python -c "import site; print(site.getsitepackages()[0])"` && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    rm -rf flash-attention/

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.1
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# TE does not have wheel on cuda 13 yet, thus need to install from source
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==26.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT}

# Pinned to d64a639 (older than slime upstream's a193d9d); the newer commit
# requires TMS_CUDA_MAJOR env var to build a multi-CUDA wheel.
RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# This patch from masahi will be included in later Triton releases
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    (cd /root && git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git && cd triton && pip install -r python/requirements.txt && pip install --verbose -e .); \
  fi

# Skip sglang lines from requirements (we install --no-deps below; the full
# transitive set conflicts with the vllm base environment).
COPY requirements.txt /tmp/requirements.txt
RUN grep -vE '^[[:space:]]*(sglang|sglang-router)([[:space:]]|[<>=!]|$)' /tmp/requirements.txt > /tmp/requirements-vllm.txt && \
    pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements-vllm.txt

# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# https://github.com/pytorch/pytorch/issues/168167
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

# vime's slime/utils/arguments.py + slime/ray/rollout.py top-level import sglang_router;
# the cu129 vllm base does not ship sglang, so install --no-deps stubs here.
RUN pip install --no-deps "sglang==${SGLANG_VERSION}" sglang-router==0.3.2 && \
    pip install IPython

# Pin vllm-router explicitly so the vllm rollout routing layer is a visible build step
# (also in requirements.txt; pulling it here makes the layer cache-able and fail-fast).
RUN pip install "vllm-router>=0.1.14"

RUN rm -rf /root/.cache/pip

# ====================================== Patches ============================================

# Patch megatron BEFORE pip install -e . so any patch hunks that touch setup.py
# or C++/CUDA extensions are picked up by the build.
COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch && \
    pip install -e .

# ====================================== Install main package ============================================

ARG VIME_COMMIT=main
RUN git clone https://github.com/vllm-project/vime.git /root/slime && \
    cd /root/slime && \
    git checkout ${VIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

# ====================================== Build-time smoke ============================================

# Fail-fast import smoke + flashinfer version pin check. Catches ABI / version
# regressions at build time instead of first GPU run.
RUN python3 -c "\
import vllm, sglang, slime, flashinfer; \
from slime.backends.vllm_utils.vllm_engine import VLLMEngine; \
print('vllm', vllm.__version__); \
print('sglang', sglang.__version__); \
print('flashinfer', flashinfer.__version__); \
print('VLLMEngine import ok')"

# Megatron stream serialization. vllm rollout + megatron actor co-located on
# the same GPU need this; default value causes stream-level contention.
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# Reset ENTRYPOINT inherited from vllm/vllm-openai base (`vllm serve`).
ENTRYPOINT []
CMD ["/bin/bash"]
