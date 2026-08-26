#!/bin/bash

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
pkill -9 -f VLLM
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
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
MODEL_PATH="${MODEL_PATH:-/home/vllm/weights/Qwen3-4B}"
PROMPT_DATA_PATH="${PROMPT_DATA_PATH:-/home/vllm/c00944022/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-/home/vllm/c00944022/0623/vime-delta-weights}"
UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR="${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:-/tmp/vime-rollout-checkpoint}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6399}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-vime-delta}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --load "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 200
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

UPDATE_WEIGHT_ARGS=(
   --update-weight-mode delta
   --update-weight-transport disk
   --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
   --update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
   --update-weight-delta-encoding xor
   --update-weight-delta-checksum xxh3-128
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

ray start --head --port="${RAY_GCS_PORT}" --temp-dir="${RAY_TEMP_DIR}" \
--node-ip-address 127.0.0.1 --disable-usage-stats \
--dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"

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
${UPDATE_WEIGHT_ARGS[@]} \
${MISC_ARGS[@]}
