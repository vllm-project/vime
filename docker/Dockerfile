ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0-cu129-ubuntu2404
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

# MEGATRON_COMMIT + PATCH_VERSION track slime upstream's docker/Dockerfile
# (slime 1dcf0da / patch latest as of 2026-02-14). vllm 0.21 cu129 base gives
# us torch 2.11.0+cu129, libcudart.so.12 natively, vllm + vllm-router pre-installed.
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG PATCH_VERSION=latest

# Pinned to match slime upstream Dockerfile for parity (sglang here is install-on-top
# via --no-deps; the cu129 base ships vllm, not sglang).
ARG SGLANG_VERSION=0.5.10.post1
ARG SLIME_TORCH_MEMORY_SAVER_COMMIT=d64a639

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

WORKDIR /root/

# ======================================== Apt dependencies =============================================

# TE 2.10 has no prebuilt wheel — always builds from source.  vllm/vllm-openai
# base is an inference image, so the cu12 dev-header collection needed by the TE
# source build (cusparse / cusolver / cufft / curand / cudnn / nvrtc / nvml /
# profiler-api / nvtx) has to be installed explicitly.  cuda-cudart-12-9 is
# already present in the base.  cmake / ninja / build-essential drive apex +
# flash-attn 2/3 source builds.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git rsync dnsutils nvtop \
      cmake ninja-build build-essential \
      cuda-nvrtc-dev-12-9 cuda-nvml-dev-12-9 cuda-profiler-api-12-9 cuda-nvtx-12-9 \
      libcusparse-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9 \
      libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# python → python3 symlink early so subsequent source builds (flash-attn hopper,
# apex) that invoke `python` succeed. slime upstream's base (slimerl/sglang) has
# this symlink already; vllm/vllm-openai base only ships python3.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ====================================== Python dependencies ============================================

# flash-attn 2.7.4.post1 — slime upstream parity. TE 2.10 doesn't support newer
# FA2 majors, so this version is the ceiling. Source compile (~3-5 min with MAX_JOBS).
RUN MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

# flash-attn 3 hopper kernels — slime upstream parity. Source compile from the
# dao-AILab/flash-attention `hopper/` subdir at the pinned commit. Installed under
# `flash_attn_3/` so callers explicitly opt in (slime does, vllm has its own FA3 path).
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && \
    git checkout fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89 && \
    git submodule update --init && \
    cd hopper/ && \
    MAX_JOBS=96 python setup.py install && \
    python_path=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p $python_path/flash_attn_3 && \
    cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py && \
    rm -rf /root/flash-attention/

# TransformerEngine 2.10 — cu12 wheel is available, no source compile.
RUN pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

# apex from source — slime upstream parity. Enables rope/grad fusion in Megatron;
# without this, launcher must pass --no-rope-fusion --no-gradient-accumulation-fusion.
# Slow (~10 min) but cached on rebuild.
RUN NVCC_APPEND_FLAGS="--threads 4" \
    pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

# Megatron-LM (clone now, patch + install after the patch COPY layer below).
RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive /root/Megatron-LM && \
    cd /root/Megatron-LM && \
    git checkout "${MEGATRON_COMMIT}"

# Apply slime's Megatron patch.
COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/megatron.patch
RUN cd /root/Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Megatron patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch && \
    pip install -e . --no-deps

# torch_memory_saver from source (slime parity) — pypi wheel ships cu12 binary so
# would work too, but slime uses the source build for newer features.
RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@${SLIME_TORCH_MEMORY_SAVER_COMMIT} --no-cache-dir --force-reinstall

# mbridge + Megatron-Bridge + modelopt — needed by slime's --megatron-to-hf-mode bridge
# checkpoint load path.
RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps && \
    pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation && \
    pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# fla + tilelang — linear-attention model variants (slime_plugins/models/qwen3_5.py,
# qwen3_next.py). cu12 native wheels here, no version skew vs base.
RUN pip install flash-linear-attention==0.4.1 && \
    pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# sglang stubs — vime's slime/utils/arguments.py + slime/ray/rollout.py still import
# sglang_router at module load. Install --no-deps so it doesn't drag a full sglang
# stack into the cu12 environment. vllm rollout doesn't actually exercise sglang code.
RUN pip install --no-deps "sglang==${SGLANG_VERSION}" sglang-router==0.3.2 && \
    pip install IPython

# pytorch issue #168167 workaround (slime upstream's pin).
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# Megatron requires numpy 1.x.
RUN pip install "numpy<2"

# Install slime runtime requirements while excluding the sglang lines (we ship the
# stub via --no-deps above; the requirements file pulls sglang's full transitive set
# which conflicts with the vllm-base environment).
COPY requirements.txt /tmp/slime-requirements.txt
RUN python3 - <<'PY' && \
    python3 -m pip install -r /tmp/slime-requirements-vllm.txt
from pathlib import Path

source = Path("/tmp/slime-requirements.txt")
target = Path("/tmp/slime-requirements-vllm.txt")
skip_prefixes = ("sglang", "sglang-router")

lines = []
for raw_line in source.read_text().splitlines():
    normalized = raw_line.split("#", 1)[0].strip().lower()
    if normalized.startswith(skip_prefixes):
        continue
    lines.append(raw_line)

target.write_text("\n".join(lines) + "\n")
PY

ENV PYTHONPATH="/root/Megatron-LM:${PYTHONPATH}"

# pytest for in-container test runs. (python symlink is set early, above.)
RUN python3 -m pip install pytest

# Install slime itself last so changes to the source don't bust earlier layers.
COPY . /root/slime
RUN cd /root/slime && \
    pip install -e . --no-deps

WORKDIR /root/slime

# Reset ENTRYPOINT inherited from vllm/vllm-openai base (which sets
# `["vllm serve"]`). Without this, every `docker run <image> <cmd>` invocation
# gets `<cmd>` interpreted as an argument to `vllm serve`.
ENTRYPOINT []
CMD ["/bin/bash"]
