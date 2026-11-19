#!/bin/bash

set -ex

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="/root/Megatron-LM/:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

DATA_ROOT="${DATA_ROOT:-/root}"

CKPT_ARGS=(
   --hf-checkpoint ${DATA_ROOT}/weights/Qwen3-30B-A3B/
   --load ${DATA_ROOT}/weights/Qwen3-30B-A3B/
   --ref-load ${DATA_ROOT}/weights/Qwen3-30B-A3B/
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3
   --rollout-batch-size 2
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 8
   --balance-data
)

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
   --entropy-coef 0.001
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
   --vllm-additional-config '{"weight_nz_mode":0}'
   --rollout-num-gpus-per-engine 2
   --vllm-gpu-memory-utilization 0.6
   --vllm-pipeline-parallel-size 1
   --vllm-enable-expert-parallel
   --vllm-expert-placement-strategy linear
   --vllm-enable-scale-out
   --vllm-enforce-eager
)

UPDATE_WEIGHT_ARGS=(
   --update-weight-mode delta
   --update-weight-transport sparse_hccl
)

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

ray start --head --num-gpus 0 --resources '{"NPU": 6}' --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 2 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${UPDATE_WEIGHT_ARGS[@]} \
   ${MISC_ARGS[@]}
