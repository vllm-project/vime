#!/usr/bin/env bash
# Prepare RULER-HQA eval JSON (eval_{50,100,200,...}.json).
#
# Usage:
#   bash examples/mem_agent/prepare-eval-data.sh
#   LENGTHS="50 200 800" bash examples/mem_agent/prepare-eval-data.sh
#   bash examples/mem_agent/prepare-eval-data.sh --download
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

DATA_DIR="${DATA_DIR:-${DATA_ROOT}}"
HF_DATASET="${HF_DATASET:-BytedTsinghua-SIA/hotpotqa}"
LENGTHS="${LENGTHS:-50 100 200 400 800 1600 3200 6400}"

mkdir -p "${DATA_DIR}"
export PYTHONPATH="/root/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"

download_one() {
  local length="$1"
  local fname="eval_${length}.json"
  local dest="${DATA_DIR}/${fname}"
  if [[ -f "${dest}" ]]; then
    echo "[SKIP] ${fname}"
    return 0
  fi
  echo "[DOWNLOAD] ${fname}"
  python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id="${HF_DATASET}", filename="${fname}", repo_type="dataset")
shutil.copy2(path, "${dest}")
print("Saved:", "${dest}")
PY
}

if [[ "${1:-}" == "--download" ]]; then
  for length in ${LENGTHS}; do
    download_one "${length}"
  done
else
  missing=0
  for length in ${LENGTHS}; do
    if [[ ! -f "${DATA_DIR}/eval_${length}.json" ]]; then
      echo "MISSING: ${DATA_DIR}/eval_${length}.json"
      missing=$((missing + 1))
    fi
  done
  if [[ "${missing}" -gt 0 ]]; then
    echo ""
    echo "Place eval_*.json under ${DATA_DIR}, or run:"
    echo "  bash examples/mem_agent/prepare-eval-data.sh --download"
    exit 1
  fi
  echo "All eval files present in ${DATA_DIR}"
  ls -lh "${DATA_DIR}"/eval_*.json 2>/dev/null | head -20
fi
