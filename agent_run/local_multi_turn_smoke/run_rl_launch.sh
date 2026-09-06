#!/usr/bin/env bash
# ROCm/MI355X (gfx950) port of run_h200.sh.
set -euo pipefail

ROOT=${ROOT:-/mnt/m2m_nobackup/lizli102/vime-agent-smoke}
mkdir -p "${ROOT}/runs"

# ROCm device passthrough replaces `--gpus all`; --group-add video + seccomp
# unconfined per the repo's AMD tutorial. docker.sock/binary are mounted so the
# per-task LocalDockerSandbox can spawn sibling containers on the host daemon.
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
  -e MODEL_DIR="${MODEL_DIR:-/work/models/Qwen3-4B}" \
  -e VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.60}" \
  -e MAX_TURNS="${MAX_TURNS:-80}" \
  -e ENT="${ENT:-0.0}" \
  -e LR="${LR:-1e-6}" \
  -e TASKS="${TASKS:-sympy-10.jsonl}" \
  -e MODEL_CONF="${MODEL_CONF:-qwen3-4B}" \
  -e CKPT_DIR="${CKPT_DIR:-/work/runs/ckpt}" -e MAX_SEQS="${MAX_SEQS:-8}" -e MAX_BT="${MAX_BT:-4096}" \
  -e GPUS="${GPUS:-0,1,2,3,4,5,6,7}" -e TP="${TP:-1}" -e NGPU="${NGPU:-8}" \
  -e NUM_ROLLOUT="${NUM_ROLLOUT:-2}" -e RB="${RB:-4}" -e N_SAMPLES="${N_SAMPLES:-4}" \
  -e RESP="${RESP:-4096}" -e GB="${GB:-16}" -e CTX="${CTX:-32768}" -e MTPG="${MTPG:-32768}" \
  -w /root/vime \
  --entrypoint bash vllm/vime-rocm:latest \
  agent_run/local_multi_turn_smoke/run_rl.sh
