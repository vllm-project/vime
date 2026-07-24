#!/bin/bash
# Tiny end-to-end fully-async GRPO example using Qwen3-4B-Instruct on the
# dapo-math-17k dataset. Designed to run on a single 4-GPU node in a few minutes
#
# Prerequisites:
#   /root/models/Qwen3-4B/             (HF checkpoint)
#   /root/datasets/dapo-math-17k/dapo-math-17k.jsonl

# clean any leftover ray/vllm
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
export RAY_USE_UVLOOP=0 # upstream: Ray's uvloop integration has caused intermittent async actor issues.
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
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh"

MODEL_DIR=${MODEL_DIR:-/root/models/Qwen3-4B}
DATA_PATH=${DATA_PATH:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --load "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}"
   --megatron-to-hf-mode bridge
   --save /tmp/vime_fully_async_demo/
   --save-interval 9999
)

ROLLOUT_ARGS=(
   # ↓↓↓ This is the only knob you need to flip to go fully-async ↓↓↓
   --rollout-function-path vime.rollout.fully_async_rollout.generate_rollout_fully_async

   --prompt-data "${DATA_PATH}"
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
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
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
   --rollout-num-gpus-per-engine 2
   --vllm-gpu-memory-utilization 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
)

ray start --head --node-ip-address 127.0.0.1 --disable-usage-stats

# fully-async splits actor / rollout onto disjoint GPUs (no colocation).
ACTOR_GPUS=2
ROLLOUT_GPUS=2

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
