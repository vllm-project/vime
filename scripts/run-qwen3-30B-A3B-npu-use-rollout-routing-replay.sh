#!/bin/bash

# for rerun the task
pkill -9 -f "vllm serve" || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
pkill -9 redis || true

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/workspace/issue205/Megatron-LM:/workspace/issue205/Megatron-Bridge/src:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B-npu.sh"

CKPT_ARGS=(
   --hf-checkpoint /home/data/weights/Qwen3-30B-A3B/
   --load /home/data/weights/Qwen3-30B-A3B/
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /home/w00893744/dataset/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 300
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len $((1024 * 8))
   --rollout-temperature 1.0
   --global-batch-size 32
   --balance-data
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
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project vime-dev
   # --wandb-group qwen3-30B-A3B-npu-routing-replay
   # --wandb-key ${WANDB_KEY}
)

VLLM_ARGS=(
   --rollout-backend vllm
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 8
   --use-rollout-routing-replay
   --vllm-weight-sync-mode native
   --vllm-gpu-memory-utilization 0.7
   --vllm-enable-sleep-mode
   --vllm-max-model-len $((1024 * 20))
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --no-gradient-accumulation-fusion
   --train-memory-margin-bytes 2147483648
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 16 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with the NPU variables required by workers.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"SLIME_SCRIPT_TRAIN_BACKEND\": \"${SLIME_SCRIPT_TRAIN_BACKEND}\",
    \"ASCEND_RT_VISIBLE_DEVICES\": \"${ASCEND_RT_VISIBLE_DEVICES}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${CUDA_DEVICE_MAX_CONNECTIONS}\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES}\",
    \"HCCL_HOST_SOCKET_PORT_RANGE\": \"${HCCL_HOST_SOCKET_PORT_RANGE}\",
    \"HCCL_NPU_SOCKET_PORT_RANGE\": \"${HCCL_NPU_SOCKET_PORT_RANGE}\",
    \"HYDRA_FULL_ERROR\": \"${HYDRA_FULL_ERROR}\",
    \"MASTER_PORT\": \"${MASTER_PORT}\",
    \"DISABLE_L2_CACHE\": \"${DISABLE_L2_CACHE}\",
    \"VLLM_ASCEND_ENABLE_NZ\": \"${VLLM_ASCEND_ENABLE_NZ}\",
    \"VLLM_USE_AOT_COMPILE\": \"${VLLM_USE_AOT_COMPILE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${REPO_ROOT}/train.py" \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
