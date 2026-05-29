#!/bin/bash
# Driver that launches one cross-host PR #66 config: docker container on the
# head host + docker container on the worker host, both with --network host.
#
# Usage:
#   HEAD_HOST=<ssh_host_for_head> WORKER_HOST=<ssh_host_for_worker> \
#   HEAD_IP=<head_lan_ip>         WORKER_IP=<worker_lan_ip> \
#   IMAGE=<vllm_r3_image>         LAN_IFACE=<lan_iface, e.g. ens7> \
#   WORKSPACE=<nfs_shared_slime_workspace> \
#   bash launch_cross_host.sh <CONFIG_ID>      # e.g. s3, s4, s5, s7, s9
#
# Both ${HEAD_HOST} and ${WORKER_HOST} must be SSH-reachable from where this
# script runs and must see ${WORKSPACE} and the runs dir over a shared NFS.
# ${HEAD_HOST} need not be the same as ${HEAD_IP} (e.g. ssh alias vs LAN IP).
set -e

CONFIG="${1:?usage: HEAD_HOST=... WORKER_HOST=... HEAD_IP=... WORKER_IP=... IMAGE=... bash $0 <CONFIG_ID>}"

: "${HEAD_HOST:?required}"
: "${WORKER_HOST:?required}"
: "${HEAD_IP:?required}"
: "${WORKER_IP:?required}"
: "${IMAGE:?required (e.g. inferactinc/public:vime-vllm-r3-latest)}"
: "${WORKSPACE:?required (NFS-shared path to slime workspace)}"

LAN_IFACE="${LAN_IFACE:-ens7}"
HF_CACHE="${HF_CACHE:-/home/aoshen/.cache/huggingface}"
MODELS_DIR="${MODELS_DIR:-/home/aoshen/models-shared}"
RUNS_ROOT="${RUNS_ROOT:-$(dirname "$WORKSPACE")}"

RUN_ID="pr66-sweep-${CONFIG}"
TEST_NAME="q30b-pr66-${CONFIG}"
RUNS_DIR="${RUNS_ROOT}/${RUN_ID}/${TEST_NAME}"
SCRIPTS_DIR="${RUNS_ROOT}/pr66-sweep/_scripts"
HEAD_CTNR="vime-pr66-${CONFIG}-head"
WORKER_CTNR="vime-pr66-${CONFIG}-worker"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage head.sh / worker.sh into a NFS-shared dir so both containers can mount them.
ssh "${HEAD_HOST}" "mkdir -p ${RUNS_DIR}/barrier && rm -f ${RUNS_DIR}/barrier/* && mkdir -p ${SCRIPTS_DIR}"
scp "${THIS_DIR}/head.sh" "${HEAD_HOST}:${SCRIPTS_DIR}/head.sh"
scp "${THIS_DIR}/worker.sh" "${HEAD_HOST}:${SCRIPTS_DIR}/worker.sh"
ssh "${HEAD_HOST}" "chmod +x ${SCRIPTS_DIR}/*.sh"

# Head: all 8 GPUs, --network host for cross-host Ray + vLLM master_addr reachability.
ssh "${HEAD_HOST}" "docker run -d --rm --name ${HEAD_CTNR} \
    --gpus all --ipc=host --shm-size=32g \
    --network host \
    --device=/dev/infiniband --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -e HF_HOME=/root/.cache/huggingface -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e PYTHONPATH=/root/Megatron-LM/ -e WANDB_DISABLED=true \
    -e PR66_CONFIG=${CONFIG} \
    -e HEAD_IP=${HEAD_IP} \
    -e SLIME_HOST_IP=${HEAD_IP} -e RAY_NODE_IP_ADDRESS=${HEAD_IP} \
    -e NCCL_SOCKET_IFNAME=${LAN_IFACE} -e NCCL_IB_DISABLE=1 -e GLOO_SOCKET_IFNAME=${LAN_IFACE} \
    -e SLIME_VLLM_HEALTH_TIMEOUT_SEC=1200 \
    -v ${WORKSPACE}:/root/slime \
    -v ${HF_CACHE}:/root/.cache/huggingface \
    -v ${MODELS_DIR}:/root/models \
    -v ${MODELS_DIR}:/root/datasets \
    -v ${RUNS_DIR}:/root/runs \
    -v ${SCRIPTS_DIR}/head.sh:/root/head.sh:ro \
    -w /root/slime ${IMAGE} bash /root/head.sh"

ssh "${WORKER_HOST}" "docker run -d --rm --name ${WORKER_CTNR} \
    --gpus all --ipc=host --shm-size=32g \
    --network host \
    --device=/dev/infiniband --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 --ulimit stack=67108864 \
    -e HF_HOME=/root/.cache/huggingface -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e PYTHONPATH=/root/Megatron-LM/ \
    -e HEAD_IP=${HEAD_IP} -e WORKER_IP=${WORKER_IP} \
    -e SLIME_HOST_IP=${WORKER_IP} -e RAY_NODE_IP_ADDRESS=${WORKER_IP} \
    -e NCCL_SOCKET_IFNAME=${LAN_IFACE} -e NCCL_IB_DISABLE=1 -e GLOO_SOCKET_IFNAME=${LAN_IFACE} \
    -v ${WORKSPACE}:/root/slime \
    -v ${HF_CACHE}:/root/.cache/huggingface \
    -v ${MODELS_DIR}:/root/models \
    -v ${MODELS_DIR}:/root/datasets \
    -v ${RUNS_DIR}:/root/runs \
    -v ${SCRIPTS_DIR}/worker.sh:/root/worker.sh:ro \
    -w /root/slime ${IMAGE} bash /root/worker.sh"

echo "=== Launched ${CONFIG}:"
echo "   head=${HEAD_CTNR} on ${HEAD_HOST} (${HEAD_IP})"
echo "   worker=${WORKER_CTNR} on ${WORKER_HOST} (${WORKER_IP})"
echo "Live: tail -F ${RUNS_DIR}/run.log    (head/test)"
echo "Live: tail -F ${RUNS_DIR}/worker.log (worker)"
