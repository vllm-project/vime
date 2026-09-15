#!/bin/bash
# Async (non-colocated) GRPO for Qwen3-8B on ROCm (gfx950 / MI350): train_async.py,
# actor and rollout on disjoint GPUs, RCCL-broadcast weight sync (no colocate IPC).

# for rerun the task
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

# Clear baked NVTE_* so Megatron sets the attn backend from --attention-backend flash.
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN

# ----- ROCm GPU visibility (same recipe as the colocate script) -----
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}
export HIP_VISIBLE_DEVICES=${VISIBLE_GPUS:-${HIP_VISIBLE_DEVICES:-0,1,2,3}}
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
IFS=',' read -r -a _vis <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${NUM_GPUS:-${#_vis[@]}}
HAS_NVLINK=0

# ----- async GPU split: actor pool + rollout pool are disjoint -----
ACTOR_GPUS=${ACTOR_GPUS:-2}                       # actor: TP=2 on 2 GPUs (DP=1)
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}   # rollout: remaining GPUs
echo "NUM_GPUS=$NUM_GPUS ACTOR_GPUS=$ACTOR_GPUS ROLLOUT_GPUS=$ROLLOUT_GPUS HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"

# ----- short validation run (override to scale) -----
NUM_ROLLOUT=${NUM_ROLLOUT:-5}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_vime_async/
   --save /root/Qwen3-8B_vime_async/
   --save-interval 100000
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 128
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
   --max-tokens-per-gpu 9216
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
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 2
   --vllm-gpu-memory-utilization 0.85
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-gradient-accumulation-fusion
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT:-vime-qwen3-8B-rocm}"
   --wandb-group "${WANDB_GROUP:-qwen3-8B-grpo-async-rocm}"
   --wandb-key "${WANDB_API_KEY:-${wandb_key}}"
   --wandb-mode online
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${VIME_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
