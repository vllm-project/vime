#!/bin/bash

set -euo pipefail

# Reproducible Disk + Delta / Disk + Full benchmark for Qwen3-4B and
# Qwen3-30B-A3B on the 12-NPU validation host.
MODEL_SIZE="${MODEL_SIZE:-4B}"
UPDATE_WEIGHT_MODE="${UPDATE_WEIGHT_MODE:-delta}"
NUM_ROLLOUT="${NUM_ROLLOUT:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
BENCHMARK_TAG="${BENCHMARK_TAG:-$(date +%Y%m%d-%H%M%S)}"

case "${UPDATE_WEIGHT_MODE}" in
  delta|full) ;;
  *) echo "UPDATE_WEIGHT_MODE must be delta or full" >&2; exit 2 ;;
esac

case "${MODEL_SIZE}" in
  4B)
    MODEL_CONFIG="qwen3-4B.sh"
    MODEL_PATH="${MODEL_PATH:-/home/vllm/weights/Qwen3-4B}"
    ACTOR_GPUS=4
    EXPERT_MODEL_PARALLEL_SIZE=1
    VLLM_MEMORY_UTILIZATION="${VLLM_MEMORY_UTILIZATION:-0.3}"
    SEQUENCE_PARALLEL_ARGS=()
    ;;
  30B)
    MODEL_CONFIG="qwen3-30B-A3B.sh"
    MODEL_PATH="${MODEL_PATH:-/home/vllm/weights/Qwen3-30B-A3B}"
    ACTOR_GPUS=8
    EXPERT_MODEL_PARALLEL_SIZE=8
    VLLM_MEMORY_UTILIZATION="${VLLM_MEMORY_UTILIZATION:-0.4}"
    SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
    ;;
  *) echo "MODEL_SIZE must be 4B or 30B" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/${MODEL_CONFIG}"

PROMPT_DATA_PATH="${PROMPT_DATA_PATH:-/home/vllm/c00944022/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-/home/vllm/weight-sync-bench/${BENCHMARK_TAG}-${MODEL_SIZE}-${UPDATE_WEIGHT_MODE}}"
UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR="${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:-/tmp/vime-bench-${MODEL_SIZE}-${UPDATE_WEIGHT_MODE}}"
BENCHMARK_LOG="${BENCHMARK_LOG:-/home/vllm/weight-sync-bench/logs/${BENCHMARK_TAG}-${MODEL_SIZE}-${UPDATE_WEIGHT_MODE}.log}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6399}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/vb-${MODEL_SIZE}-${UPDATE_WEIGHT_MODE}}"
VIME_WORKSPACE_ROOT="${VIME_WORKSPACE_ROOT:-/home/vllm/c00944022/0623}"

mkdir -p "$(dirname -- "${BENCHMARK_LOG}")" "${UPDATE_WEIGHT_DISK_DIR}"

cleanup_ray() {
  ray stop --force >/dev/null 2>&1 || true
  pkill -9 -f 'VLLM::' >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

# These scripts are used on an exclusive validation host. Start from a clean
# Ray/vLLM state so stale workers cannot consume NPU memory or skew timings.
pkill -9 -f '[v]llm serve|VLL[M]::' || true
pkill -9 -f VLLM || true
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
pkill -9 redis || true
sleep 3

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="${VIME_WORKSPACE_ROOT}/Megatron-Bridge/src:${VIME_WORKSPACE_ROOT}/Megatron-LM:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

UPDATE_WEIGHT_ARGS=(
  --update-weight-mode "${UPDATE_WEIGHT_MODE}"
  --update-weight-transport disk
  --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
  --update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
)
if [[ "${UPDATE_WEIGHT_MODE}" == delta ]]; then
  UPDATE_WEIGHT_ARGS+=(
    --update-weight-delta-encoding xor
    --update-weight-delta-checksum xxh3-128
  )
fi

ray start --head --port="${RAY_GCS_PORT}" --temp-dir="${RAY_TEMP_DIR}" \
  --node-ip-address 127.0.0.1 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json '{"env_vars":{"PYTORCH_NPU_ALLOC_CONF":"max_split_size_mb:128","VLLM_VERSION":"0.26.0"}}' \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus 4 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_PATH}" \
  --load "${MODEL_PATH}" \
  --ref-load "${MODEL_PATH}" \
  --megatron-to-hf-mode bridge \
  --prompt-data "${PROMPT_DATA_PATH}" \
  --input-key prompt --label-key label --apply-chat-template --rollout-shuffle \
  --rm-type math \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
  --rollout-temperature 1 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --balance-data \
  --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
  --adam-beta1 0.9 --adam-beta2 0.98 \
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --advantage-estimator grpo --kl-loss-coef 0.0 --kl-loss-type low_var_kl \
  --kl-coef 0.0 --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
  --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}" \
  --expert-tensor-parallel-size 1 \
  "${SEQUENCE_PARALLEL_ARGS[@]}" \
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
  --use-dynamic-batch-size --max-tokens-per-gpu 8192 \
  --rollout-num-gpus-per-engine 4 \
  --vllm-gpu-memory-utilization "${VLLM_MEMORY_UTILIZATION}" \
  --vllm-enforce-eager \
  "${UPDATE_WEIGHT_ARGS[@]}" \
  --attention-dropout 0.0 --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
  --attention-backend flash --micro-batch-size 1 --use-flash-attn \
  2>&1 | tee "${BENCHMARK_LOG}"

# A completed transport benchmark is invalid when training did not mutate the
# model. This catches truncated/all-zero-reward runs before their timings are
# reported as real Delta results.
if ! grep -Eq "'train/grad_norm': (0\.[0-9]*[1-9]|[1-9][0-9]*\.?[0-9]*|[1-9]\.[0-9]*e[-+]?[0-9]+)" "${BENCHMARK_LOG}"; then
  echo "invalid benchmark: no non-zero train/grad_norm in ${BENCHMARK_LOG}" >&2
  exit 3
fi
if [[ "${UPDATE_WEIGHT_MODE}" == delta ]] && grep -q 'density=0.00% wire=0.00 GB' "${BENCHMARK_LOG}"; then
  echo "invalid benchmark: at least one Delta round has an empty payload" >&2
  exit 4
fi

echo "validated benchmark log: ${BENCHMARK_LOG}"
