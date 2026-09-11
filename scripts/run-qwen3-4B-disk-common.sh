#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
: "${ASCEND_RT_VISIBLE_DEVICES:?Set ASCEND_RT_VISIBLE_DEVICES to the allocated NPUs}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

DATA_ROOT="${DATA_ROOT:-/root}"

CKPT_ARGS=(
   --hf-checkpoint ${DATA_ROOT}/models/Qwen3-4B/
   --load ${DATA_ROOT}/models/Qwen3-4B/
   --ref-load ${DATA_ROOT}/models/Qwen3-4B/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT:-3}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --megatron-to-hf-mode bridge
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.0
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
   --vllm-gpu-memory-utilization 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --micro-batch-size 1
   --use-flash-attn
)

SYNC_ARGS=(--update-weight-mode "${UPDATE_WEIGHT_MODE}" --update-weight-transport disk
   --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}")
if [[ "${UPDATE_WEIGHT_MODE}" == delta ]]; then
   SYNC_ARGS+=(--update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
      --update-weight-delta-encoding "${UPDATE_WEIGHT_DELTA_ENCODING:-xor}")
fi
cd "${SCRIPT_DIR}/.."
ray start --head --port="${RAY_GCS_PORT}" --temp-dir="${RAY_TEMP_DIR}" --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
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
${VLLM_ARGS[@]} \
"${MISC_ARGS[@]}" \
"${SYNC_ARGS[@]}" \
"$@"
