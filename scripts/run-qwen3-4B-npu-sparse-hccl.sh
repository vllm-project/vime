#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export UPDATE_WEIGHT_MODE=sparse
export UPDATE_WEIGHT_TRANSPORT=nccl
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"

exec bash "${SCRIPT_DIR}/run-qwen3-4B-npu.sh" "$@"
