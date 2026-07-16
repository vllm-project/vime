#!/bin/bash

# Single-turn Qwen3-VL GRPO on geo3k for Ascend NPU.
set -eo pipefail

MODEL_NAME=${VIME_SCRIPT_MODEL_NAME:-Qwen3-VL-8B-Instruct}
DATASET_NAME=${VIME_SCRIPT_DATASET_NAME:-chenhegu/geo3k_imgurl}
NUM_GPUS=${VIME_SCRIPT_NUM_GPUS:-8}
NUM_ROLLOUT=${VIME_SCRIPT_NUM_ROLLOUT:-3000}
DATA_ROOT="/root/datasets/$(basename "$DATASET_NAME")"
MODEL_ROOT="/root/models/${MODEL_NAME}"

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$NUM_ROLLOUT" =~ ^[1-9][0-9]*$ ]]; then
   echo "Error: VIME_SCRIPT_NUM_GPUS and VIME_SCRIPT_NUM_ROLLOUT must be positive integers"
   exit 1
fi

VALID_MODELS="
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: unsupported MODEL_NAME=${MODEL_NAME}"
   exit 1
fi

if [ -z "${VIME_SCRIPT_EXTERNAL_RAY:-}" ] || [ "$VIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/root/Megatron-Bridge/src:/root/Megatron-LM/:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest/
export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/
export ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest/
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest/
export set_env_path=/usr/local/Ascend/nnal/atb/set_env.sh

IFS=',' read -r -a VISIBLE_NPUS <<<"${ASCEND_RT_VISIBLE_DEVICES}"
REQUIRED_NPUS=$((NUM_GPUS * 2))
if [ "${#VISIBLE_NPUS[@]}" -lt "$REQUIRED_NPUS" ]; then
   echo "Error: actor and rollout require ${REQUIRED_NPUS} visible NPUs"
   exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p /root/models /root/datasets
if [ ! -d "$MODEL_ROOT" ]; then
   hf download "Qwen/${MODEL_NAME}" --local-dir "$MODEL_ROOT"
fi
if [ ! -d "$DATA_ROOT" ]; then
   hf download --repo-type dataset "$DATASET_NAME" --local-dir "$DATA_ROOT"
fi

MODEL_LOAD_ARGS=(
   --hf-checkpoint "$MODEL_ROOT"
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_ROOT}/train.parquet"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --global-batch-size 512
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
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
   --vllm-max-model-len 16384
   --vllm-generation-config auto
   --vllm-logprobs-mode processed_logprobs
)

if [ -n "${WANDB_API_KEY:-}" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project vime-geo3k-vlm
      --wandb-group "${MODEL_NAME,,}-megatron-vllm-${NUM_GPUS}npu"
      --wandb-key "$WANDB_API_KEY"
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

BACKEND_ARGS=(
   --train-backend megatron
   --load "$MODEL_ROOT"
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

VIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${VIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

pkill -9 -f '[v]llm serve|VLL[M]::' || true
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force || true
   pkill -9 ray || true
fi
pkill -9 vime || true
pkill -9 redis || true

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray start --head --node-ip-address "$MASTER_ADDR" --disable-usage-stats \
      --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_KEYS=(
   CUDA_DEVICE_MAX_CONNECTIONS
   RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
   PYTHONPATH
   PYTORCH_ALLOC_CONF
   PYTORCH_NPU_ALLOC_CONF
   VLLM_ASCEND_ENABLE_NZ
   ASCEND_TOOLKIT_HOME
   ASCEND_OPP_PATH
   ASCEND_AICPU_PATH
   ASCEND_HOME_PATH
   set_env_path
   HYDRA_FULL_ERROR
   HCCL_HOST_SOCKET_PORT_RANGE
   HCCL_NPU_SOCKET_PORT_RANGE
   no_proxy
   MASTER_ADDR
)
RUNTIME_ENV_JSON=$(python - "${RUNTIME_ENV_KEYS[@]}" <<'PY'
import json
import os
import sys

print(json.dumps({"env_vars": {key: os.environ[key] for key in sys.argv[1:]}}))
PY
)

ray job submit --address=http://127.0.0.1:8265 \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- "$(command -v python)" train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$NUM_GPUS" \
   --rollout-num-gpus "$NUM_GPUS" \
   --multimodal-keys '{"image": "images"}' \
   "${MODEL_ARGS[@]}" \
   "${MODEL_LOAD_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${VLLM_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   --no-gradient-accumulation-fusion \
   --use-flash-attn
