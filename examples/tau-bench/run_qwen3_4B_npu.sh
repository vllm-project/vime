#!/bin/bash

if grep -q $'\r' "$0" 2>/dev/null; then
  exec bash <(sed 's/\r$//' "$0") "$@"
fi

# for rerun the task
pkill -9 vllm 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'python3 train.py' 2>/dev/null || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export VIME_VLLM_SERVER_HEALTH_TIMEOUT_SEC=900

unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

export PYTHONPATH="${SCRIPT_DIR}:/root/Megatron-Bridge/src:/root/Megatron-LM:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/root}"
TAU_BENCH_ROOT="${TAU_BENCH_ROOT:-/root/tau-bench}"

CKPT_ARGS=(
   --hf-checkpoint ${DATA_ROOT}/weights/Qwen3-4B-Instruct-2507/
   --load ${DATA_ROOT}/weights/Qwen3-4B-Instruct-2507/
   --ref-load ${DATA_ROOT}/weights/Qwen3-4B-Instruct-2507/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data ${TAU_BENCH_ROOT}/retail_train_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-max-context-len 16384
   --rollout-temperature 0.7
   --global-batch-size 256
   --dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data retail-dev ${TAU_BENCH_ROOT}/retail_dev_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.01
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1
   --vllm-gpu-memory-utilization 0.7
   --vllm-max-model-len 16384
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_tau.generate
   --custom-rm-path generate_with_tau.batched_tau_bench_rm
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${VLLM_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${MISC_ARGS[@]}"

