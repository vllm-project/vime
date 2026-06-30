#!/usr/bin/env bash
# Convert MemAgent Megatron checkpoints to HuggingFace format.
#
# Usage:
#   bash examples/mem_agent/convert-to-hf.sh
#   CHECKPOINT_DIR=/path/to/ckpt SINGLE_ITER=iter_0000199 bash examples/mem_agent/convert-to-hf.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SAVE_PATH}}"
OUTPUT_BASE="${OUTPUT_BASE:-${CHECKPOINT_DIR}-HF}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/root/Megatron-LM}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-${VIME_ROOT}/tools/convert_torch_dist_to_hf.py}"

mkdir -p "${OUTPUT_BASE}"
export PYTHONPATH="${MEGATRON_LM_DIR}:${VIME_ROOT}:${PYTHONPATH:-}"

if [[ -n "${SINGLE_ITER:-}" ]]; then
  ITERS=("${CHECKPOINT_DIR}/${SINGLE_ITER}")
else
  mapfile -t ITERS < <(ls -d "${CHECKPOINT_DIR}"/iter_* 2>/dev/null | sort)
fi

if [[ ${#ITERS[@]} -eq 0 ]]; then
  echo "ERROR: no iter_* checkpoints in ${CHECKPOINT_DIR}"
  exit 1
fi

echo "Converting ${#ITERS[@]} checkpoint(s) -> ${OUTPUT_BASE}"
FAILED=()

for iter_path in "${ITERS[@]}"; do
  iter_name="$(basename "${iter_path}")"
  output_dir="${OUTPUT_BASE}/${iter_name}"

  if [[ -d "${output_dir}" && -f "${output_dir}/config.json" ]]; then
    echo "[SKIP] ${iter_name} already at ${output_dir}"
    continue
  fi

  echo "[CONVERT] ${iter_name} -> ${output_dir}"
  if python3 "${CONVERT_SCRIPT}" \
      --input-dir "${iter_path}" \
      --output-dir "${output_dir}" \
      --origin-hf-dir "${ORIGIN_HF_DIR}"; then
    echo "[DONE] ${iter_name}"
  else
    echo "[FAILED] ${iter_name}"
    FAILED+=("${iter_name}")
  fi
done

echo "===== Summary: total=${#ITERS[@]} failed=${#FAILED[@]} ====="
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "Output: ${OUTPUT_BASE}"
