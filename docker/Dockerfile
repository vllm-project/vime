ARG BASE_IMAGE=vllm/vllm-openai:v0.22.0-cu129-ubuntu2404
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087

ARG ENABLE_CUDA_13=0
ARG TMS_CUDA_MAJOR=

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

# cublas + (cu13) cuda dev headers. The arm64 (sbsa) vllm/vllm-openai base ships
# the cublas runtime .so but not the dev header (cublas_v2.h) / unversioned .so
# symlink that the x86 base has; TE needs the header and its CMake only creates
# the CUDA::cublas target when both exist. The dev pkg must match the toolkit
# CUDA major. For cu13 the top apt block's -12-9 dev set is the wrong major, so
# also install the -13-0 nvrtc/nvtx/etc. headers TE's nvcc build needs. Placed
# after the slow flash-attn layers so their cache is preserved; before TE.
RUN apt-get update && \
    if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      apt-get install -y libcublas-dev-13-0 \
        cuda-nvrtc-dev-13-0 cuda-nvtx-13-0 cuda-nvml-dev-13-0 cuda-profiler-api-13-0 \
        libcusparse-dev-13-0 libcusolver-dev-13-0 libcufft-dev-13-0 libcurand-dev-13-0; \
    else apt-get install -y libcublas-dev-12-9; fi && \
    rm -rf /var/lib/apt/lists/*

# TE does not have wheel on cuda 13 yet, thus need to install from source.
# TE is built with --no-build-isolation, so its build deps must be pre-installed;
# install the set TE's release_v2.10 CI uses (wheel packaging ninja pybind11).
# nvidia-mathdx is left unpinned (as in TE CI): 26.6.0 is x86-only, arm64 (sbsa)
# max is 25.6.0, and TE release_v2.10 does not pin or reference it (it links plain
# CUDA::cublas), so the resolver picking the per-arch latest is safe.
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx pybind11 ninja wheel packaging && \
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

# torch_memory_saver pinned to a193d9dd (upstream slime #1916). The newer commit
# ships a multi-CUDA wheel and requires TMS_CUDA_MAJOR at build time; default it
# to the running torch's CUDA major (slime #1924).
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# This patch from masahi will be included in later Triton releases
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    (cd /root && git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git && cd triton && pip install -r python/requirements.txt && pip install --verbose -e .); \
  fi

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# https://github.com/pytorch/pytorch/issues/168167
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

RUN pip install IPython

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

COPY docker/patch/${PATCH_VERSION}/vllm.patch /tmp/
RUN VLLM_SITE=$(python3 -c "import vllm; print(vllm.__file__.rsplit('/',1)[0])") && \
    cd ${VLLM_SITE} && \
    patch -p2 --no-backup-if-mismatch < /tmp/vllm.patch && \
    rm /tmp/vllm.patch

# ====================================== Install main package ============================================

ARG VIME_COMMIT=main
RUN git clone https://github.com/vllm-project/vime.git /root/vime && \
    cd /root/vime && \
    git checkout ${VIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/vime/vime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

# ====================================== Build-time smoke ============================================

# Fail-fast import smoke + flashinfer version pin check. Catches ABI / version
# regressions at build time instead of first GPU run.
RUN python3 -c "\
import vllm, vime, flashinfer; \
from vime.backends.vllm_utils.vllm_engine import VLLMEngine; \
print('vllm', vllm.__version__); \
print('flashinfer', flashinfer.__version__); \
print('VLLMEngine import ok')"

# Megatron stream serialization. vllm rollout + megatron actor co-located on
# the same GPU need this; default value causes stream-level contention.
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# Reset ENTRYPOINT inherited from vllm/vllm-openai base (`vllm serve`).
ENTRYPOINT []
CMD ["/bin/bash"]
