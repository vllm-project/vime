#!/bin/bash

# Isolated validation variant: it does not stop processes outside this run.
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-64000-64050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-64100-64150}"
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
# Keep snapshots on the host to leave NPU memory available for training.
export VIME_SPARSE_HCCL_SNAPSHOT_DEVICE="${VIME_SPARSE_HCCL_SNAPSHOT_DEVICE:-cpu}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VIME_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${VIME_REPO_ROOT}"
export PYTHONPATH="${VIME_REPO_ROOT}:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to Qwen3-30B-A3B weights}"
PROMPT_DATA_PATH="${PROMPT_DATA_PATH:?set PROMPT_DATA_PATH to the prompt JSONL}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --load "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

# The short weight-sync validation is below the original eval interval (20),
# so no evaluation is scheduled and no external DATA_ROOT is required.
EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 2
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef "${ENTROPY_COEF:-0.001}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 4
   --vllm-pipeline-parallel-size 2
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
   --vllm-enforce-eager
   --vllm-enable-expert-parallel
   --vllm-expert-placement-strategy linear
   --vllm-additional-config '{"weight_nz_mode":0}'
)

UPDATE_WEIGHT_ARGS=(
   --update-weight-delta-batch-diff "${UPDATE_WEIGHT_DELTA_BATCH_DIFF:-32}"
   --update-weight-delta-batch-gather "${UPDATE_WEIGHT_DELTA_BATCH_GATHER:-32}"
   --update-weight-mode "${UPDATE_WEIGHT_MODE:-delta}"
   --update-weight-transport "${UPDATE_WEIGHT_TRANSPORT:-sparse_hccl}"
   --update-weight-delta-verify-every "${UPDATE_WEIGHT_DELTA_VERIFY_EVERY:-1}"
)
if [[ "${UPDATE_WEIGHT_TRANSPORT:-sparse_hccl}" == "disk" ]]; then
   UPDATE_WEIGHT_ARGS+=(--update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR:?set UPDATE_WEIGHT_DISK_DIR}")
fi
if [[ "${UPDATE_WEIGHT_MODE:-delta}" == "delta" && "${UPDATE_WEIGHT_TRANSPORT:-sparse_hccl}" == "disk" ]]; then
   UPDATE_WEIGHT_ARGS+=(
      --update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:?set UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
      --update-weight-delta-encoding xor
      --update-weight-delta-checksum xxh3-128
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --micro-batch-size 1

   --use-flash-attn
   --no-gradient-accumulation-fusion
)

RESULT_LOG="${RESULT_LOG:-qwen3-30b-tp2-pp2-ep2-sparse-hccl.log}"
if ! ray status --address="127.0.0.1:${RAY_GCS_PORT:-6523}" >/dev/null 2>&1; then
ray start --head --port="${RAY_GCS_PORT:-6523}" --temp-dir="${RAY_TEMP_DIR:-/tmp/vime-tp2-pp2-ep2}" \
   --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_DASHBOARD_PORT:-8393}" \
   --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT:-52386}"
fi

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8393}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
${VLLM_ARGS[@]} \
${UPDATE_WEIGHT_ARGS[@]} \
${MISC_ARGS[@]} 2>&1 | tee "${RESULT_LOG}"
