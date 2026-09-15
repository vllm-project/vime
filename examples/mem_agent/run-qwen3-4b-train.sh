#!/usr/bin/env bash
# MemAgent GRPO training — Qwen3-4B, vime + vLLM colocate.
#
# Usage:
#   bash examples/mem_agent/run-qwen3-4b-train.sh
#   NUM_ROLLOUT=200 SAVE_PATH=/path/to/save bash examples/mem_agent/run-qwen3-4b-train.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

echo "=== MemAgent train NUM_ROLLOUT=${NUM_ROLLOUT} SAVE_PATH=${SAVE_PATH} ==="
if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "ERROR: training data not found: ${TRAIN_DATA}"
  exit 1
fi

mem_agent_detect_nvlink
mem_agent_detect_gpus
mem_agent_cleanup
mem_agent_launch_train

echo "=== MemAgent train finished ==="
