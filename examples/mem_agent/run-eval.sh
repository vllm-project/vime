#!/usr/bin/env bash
# MemAgent RULER-HQA evaluation via vLLM serve + eval_ruler_hqa.py.
#
# Usage:
#   MODEL_PATH=/path/to/hf_ckpt bash examples/mem_agent/run-eval.sh
#   CONVERT=1 SINGLE_ITER=iter_0000199 bash examples/mem_agent/run-eval.sh
#   MODEL_PATH=/root/Qwen3-4B SAVE_FILE=Qwen3-4B-base bash examples/mem_agent/run-eval.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

EVAL_PY="${MEM_AGENT_DIR}/eval_ruler_hqa.py"
TP="${TP:-1}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"
LENGTH="${LENGTH:-50 200 800}"
SAVE_DIR="${SAVE_DIR:-${MEM_AGENT_DIR}/results}"
API="${API:-recurrent}"
N_PROC="${N_PROC:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
FORCE="${FORCE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SAVE_PATH}}"
SINGLE_ITER="${SINGLE_ITER:-iter_0000199}"
CONVERT="${CONVERT:-0}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SAVE_DIR}/eval_${STAMP}.log"
mkdir -p "${SAVE_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== MemAgent eval started at $(date) ==="
echo "LOG_FILE=${LOG_FILE}"

if [[ "${CONVERT}" == "1" ]]; then
  echo "[step] Converting ${SINGLE_ITER} ..."
  SINGLE_ITER="${SINGLE_ITER}" CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    bash "${MEM_AGENT_DIR}/convert-to-hf.sh"
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_DIR}-HF/${SINGLE_ITER}}"
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "ERROR: MODEL_PATH is required."
  echo "  MODEL_PATH=/path/to/hf_ckpt bash examples/mem_agent/run-eval.sh"
  echo "  CONVERT=1 bash examples/mem_agent/run-eval.sh"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

SAVE_FILE="${SAVE_FILE:-$(basename "${MODEL_PATH%/}")}"
MODEL_NAME="${MODEL_PATH}"

LENGTHS="${LENGTH}" bash "${MEM_AGENT_DIR}/prepare-eval-data.sh"

export PYTHONPATH="/root/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"
export VLLM_SERVE_HOST="${SERVE_HOST}" VLLM_SERVE_PORT="${SERVE_PORT}"
export SERVE_HOST SERVE_PORT
export DATAROOT="${DATA_ROOT}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
  local url="http://${SERVE_HOST}:${SERVE_PORT}/v1/models"
  log "Waiting for vLLM at ${url} ..."
  local attempts=0
  local max_attempts=120
  while true; do
    if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      log "ERROR: vLLM process (PID ${VLLM_PID}) died. See ${VLLM_LOG:-server log}."
      exit 1
    fi
    resp=$(curl -sf --max-time 10 "${url}" 2>/dev/null || true)
    if echo "${resp}" | grep -Fq "${MODEL_NAME}" 2>/dev/null; then
      log "vLLM ready."
      break
    fi
    attempts=$((attempts + 1))
    if (( attempts >= max_attempts )); then
      log "ERROR: vLLM not ready after ${max_attempts} attempts."
      exit 1
    fi
    if (( attempts % 6 == 0 )); then
      found=$(echo "${resp}" | grep -o '"id":"[^"]*"' 2>/dev/null | head -3 || echo "(no response)")
      log "Still waiting... models: ${found}"
    fi
    sleep 5
  done
}

kill_server() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    log "Stopping vLLM (pid=${VLLM_PID}) ..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}

build_common_args() {
  local extra=()
  extra+=(--model "${MODEL_NAME}")
  extra+=(--tokenizer "${MODEL_PATH}")
  extra+=(--api "${API}")
  extra+=(--n-proc "${N_PROC}")
  extra+=(--temperature "${TEMPERATURE}")
  extra+=(--top-p "${TOP_P}")
  if [[ "${FORCE}" == "1" ]]; then
    extra+=(--force)
  fi
  echo "${extra[@]}"
}

run_hqa() {
  local common
  read -ra common <<< "$(build_common_args)"
  for length in ${LENGTH}; do
    local subdir="${SAVE_DIR}/ruler_hqa_${length}"
    log "==> ruler_hqa n_docs=${length}"
    python3 "${EVAL_PY}" \
      "${common[@]}" \
      --length "${length}" \
      --data-root "${DATA_ROOT}" \
      --save-dir "${subdir}" \
      --save-file "${SAVE_FILE}"
  done
}

log "MODEL_PATH=${MODEL_PATH}"
log "TP=${TP}  LENGTH=${LENGTH}  N_PROC=${N_PROC}"
log "DATA_ROOT=${DATA_ROOT}  SAVE_DIR=${SAVE_DIR}"

pkill -9 -f '[v]llm serve' 2>/dev/null || true
sleep 2

VLLM_LOG="${SAVE_DIR}/vllm_server_${STAMP}.log"
log "Starting vLLM serve (tp=${TP}, port=${SERVE_PORT}, max_len=${MAX_MODEL_LEN}) ..."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  vllm serve "${MODEL_PATH}" \
    --tensor-parallel-size "${TP}" \
    --host "${SERVE_HOST}" \
    --port "${SERVE_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill_server' EXIT INT TERM

wait_for_server
run_hqa

log "=== Eval finished. Results: ${SAVE_DIR} ==="
