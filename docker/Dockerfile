ARG BASE_IMAGE=vllm/vllm-openai:v0.23.0-cu129-ubuntu2404
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

# vllm/vllm-openai base is an inference image — add cu12 dev headers + cmake/git
# so TE / apex / flash-attn source builds find cusparse.h etc.
RUN apt-get update && apt-get install -y \
    nvtop rsync dnsutils prometheus git cmake \
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
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
ARG INSTALL_FLASHQLA=0
RUN if [ "${INSTALL_FLASHQLA}" = "1" ]; then \
      pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation; \
    else \
      echo "Skipping FlashQLA (INSTALL_FLASHQLA=0; sm90/Hopper-only — use --qwen-gdn-backend fla)"; \
    fi
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# cublas dev header for TE CMake (arm64 base ships runtime .so but not the header).
RUN apt-get update && apt-get install -y libcublas-dev-12-9 && rm -rf /var/lib/apt/lists/*

RUN pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT}

# torch_memory_saver pinned to a193d9dd (upstream slime #1916).
# TMS_CUDA_MAJOR is required by this pin's build backend for CUDA wheels; base is cu129 -> 12.
RUN TMS_CUDA_MAJOR=12 pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# https://github.com/pytorch/pytorch/issues/168167
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# reinstall numpy 1.x for megatron; pin scipy<1.18 alongside it. vime's vllm/vllm-openai
# base ships NO scipy, so unpinned it pulls scipy>=1.18, which hard-requires numpy>=2 and
# uses np.long (removed in numpy>=1.24) -> AttributeError against the numpy<2 above ->
# `import scipy/transformers` crash. slime's sglang base resolves scipy to 1.17.1 (numpy-1.x
# compatible) natively, so slime needs no scipy pin -- this is a vime base-image divergence.
# Real boundary is 1.18 (slime runs 1.17.1 fine), not the earlier 1.14 guess.
RUN pip install "numpy<2" "scipy<1.18"

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

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

# Patch vLLM: skip execute_dummy_batch while the engine is asleep / being put to sleep, so a
# colocate DP+EP rollout's staged sleep/wake weight-sync doesn't run a forward against freed KV
# (illegal memory access at update_weights / the post-rollout sleep offload). vLLM PR #44483
# lineage, broadened to guard on EngineCore.is_sleeping() (covers the cuMem offload window).
# vLLM is a pip install (not a git checkout) so apply with plain `git apply` (no --3way).
COPY docker/patch/${PATCH_VERSION}/vllm.patch /tmp/vllm.patch
RUN VLLM_SITE="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" && \
    cd "$VLLM_SITE" && \
    git apply -v /tmp/vllm.patch && \
    rm /tmp/vllm.patch

# ====================================== Install main package ============================================

ARG VIME_COMMIT=main
RUN git clone https://github.com/vllm-project/vime.git /root/vime && \
    cd /root/vime && \
    git checkout ${VIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/vime/vime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

# Reset ENTRYPOINT inherited from the vllm/vllm-openai base (`vllm serve`), so the
# image is a plain bash/ray environment. Without this, `docker run ... bash -c ...`
# and `ray job submit` append to `vllm serve` and break. Base-image-coupled: slime's
# base has no such entrypoint, so slime's Dockerfile doesn't need this.
ENTRYPOINT []
CMD ["/bin/bash"]
