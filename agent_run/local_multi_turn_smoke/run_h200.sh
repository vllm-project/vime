#!/usr/bin/env bash

set -euo pipefail

ROOT=/mnt/data/vime-agent-smoke
mkdir -p "${ROOT}/runs"

docker run --rm --gpus all --ipc=host --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker:ro \
  -v "${ROOT}/vime:/root/vime" \
  -v "${ROOT}/models:/work/models" \
  -v "${ROOT}/assets:/work/assets" \
  -v "${ROOT}/tasks:/work/tasks:ro" \
  -v "${ROOT}/runs:/work/runs" \
  -w /root/vime \
  vllm/vime:latest \
  bash agent_run/local_multi_turn_smoke/run_inside.sh
