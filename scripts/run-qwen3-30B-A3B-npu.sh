#!/bin/bash

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
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
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
# DISABLE_L2_CACHE=1 turns off the CANN graph-engine L2 cache for the whole
# process tree. Without it the 10-step steady state on Ascend910 runs
# ~5.5% faster (9912 -> 10456 train tok/s) with identical rollout
# throughput, so keep the runtime default (L2 cache on).
# Non-colocate runs have no torch_memory_saver conflict, so expandable
# segments reduce allocator fragmentation for the larger microbatches.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
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
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len $((1024 * 8))
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_ROOT}/datasets/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # 30720-token microbatches fill the dedicated 64GB cards (TP4+SP+EP8,
   # full recompute): 20480 -> 30720 raised train tok/s 9912 -> 10489
   # (+5.8%) with expandable_segments on the same topology.
   --max-tokens-per-gpu 30720
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
   --vllm-additional-config '{"weight_nz_mode":0}'
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.7
   --vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256)
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

ray start --head --num-gpus 0 --resources '{"NPU": 16}' --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
