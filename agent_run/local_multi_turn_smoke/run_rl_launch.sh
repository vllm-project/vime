#!/usr/bin/env bash
# ROCm/MI355X (gfx950) port of run_h200.sh.
set -euo pipefail

ROOT=${ROOT:-/mnt/m2m_nobackup/lizli102/vime-agent-smoke}
mkdir -p "${ROOT}/runs"

# ROCm device passthrough replaces `--gpus all`; --group-add video + seccomp
# unconfined per the repo's AMD tutorial. docker.sock/binary are mounted so the
# per-task LocalDockerSandbox can spawn sibling containers on the host daemon.
# Defaults live in run_rl.sh only. Forwarding "${VAR}" (rather than
# "${VAR:-default}") passes an empty string when unset, which run_rl.sh's
# ${VAR:-default} then falls back on -- so there is exactly one place to change
# a default. Duplicating them here silently shadows the script's values.
docker run -d --name vime-rl \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ulimit nofile=1048576:1048576 \
  --ipc=host --network host --shm-size 32G \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker:ro \
  -v /home/lizli102/vime:/root/vime \
  -v "${ROOT}/models:/work/models" \
  -v "${ROOT}/assets:/work/assets" \
  -v "${ROOT}/tasks:/work/tasks:ro" \
  -v "${ROOT}/runs:/work/runs" \
  -v /home/lizli102:/host \
  -e MODEL_DIR="${MODEL_DIR-}" \
  -e VLLM_MEM_UTIL="${VLLM_MEM_UTIL-}" \
  -e MAX_TURNS="${MAX_TURNS-}" \
  -e ENT="${ENT-}" \
  -e LR="${LR-}" \
  -e TASKS="${TASKS-}" \
  -e MODEL_CONF="${MODEL_CONF-}" \
  -e CKPT_DIR="${CKPT_DIR-}" -e MAX_SEQS="${MAX_SEQS-}" -e MAX_BT="${MAX_BT-}" \
  -e GPUS="${GPUS-}" -e TP="${TP-}" -e NGPU="${NGPU-}" \
  -e NUM_ROLLOUT="${NUM_ROLLOUT-}" -e RB="${RB-}" -e N_SAMPLES="${N_SAMPLES-}" \
  -e RESP="${RESP-}" -e GB="${GB-}" -e CTX="${CTX-}" -e MTPG="${MTPG-}" \
  -w /root/vime \
  --entrypoint bash vllm/vime-rocm:latest \
  agent_run/local_multi_turn_smoke/run_rl.sh
