ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0-cu129-ubuntu2404
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG SGLANG_VERSION=0.5.10.post1

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

# vllm/vllm-openai base is an inference image — add cu12 dev headers + cmake/git
# so TE / apex / flash-attn source builds find cusparse.h etc.
RUN apt update
RUN apt install -y nvtop rsync dnsutils git cmake \
    cuda-nvrtc-dev-12-9 cuda-nvml-dev-12-9 cuda-profiler-api-12-9 cuda-nvtx-12-9 \
    libcusparse-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9 \
    libcudnn9-dev-cuda-12

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

RUN pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    pip install -e .

RUN pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# vime's slime/utils/arguments.py + slime/ray/rollout.py top-level import sglang_router.
# Install --no-deps so it doesn't pull a full sglang stack into the vllm environment.
RUN pip install --no-deps "sglang==${SGLANG_VERSION}" sglang-router==0.3.2 && \
    pip install IPython

# https://github.com/pytorch/pytorch/issues/168167
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

# Skip sglang lines (we install --no-deps above; the full transitive set conflicts).
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

RUN rm -rf /root/.cache/pip

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron.patch

# ====================================== Install main package ============================================

COPY . /root/slime
RUN cd /root/slime && \
    pip install -e . --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

WORKDIR /root/slime

# Reset ENTRYPOINT inherited from vllm/vllm-openai base (`vllm serve`).
ENTRYPOINT []
CMD ["/bin/bash"]
