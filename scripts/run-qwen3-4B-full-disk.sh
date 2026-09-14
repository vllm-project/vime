#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export UPDATE_WEIGHT_MODE=full
# This directory must be visible at the same path to trainer and rollout hosts.
export UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-/tmp/vime-full-weights}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6400}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8268}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-vime-full-disk}"
exec bash "${SCRIPT_DIR}/run-qwen3-4B-disk-common.sh" "$@"
