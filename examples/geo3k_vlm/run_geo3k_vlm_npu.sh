#!/bin/bash

# Single-turn Qwen3-VL GRPO on geo3k for Ascend NPU.
# Self-contained: mirrors run_geo3k_vlm.sh but inlines the NPU handling that the
# multi-turn NPU example gets via execute_train_npu() (Ascend toolkit env, HCCL
# ports, sglang/slime cleanup, ray start without --num-gpus). No .py orchestrator.
# Usage:
#   VIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct ./run_geo3k_vlm_npu.sh

# Configuration
TRAIN_BACKEND="megatron"
MODEL_NAME=${VIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-8B-Instruct"}
DATASET_NAME=${VIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${VIME_SCRIPT_NUM_GPUS:-8}
DATA_ROOT="/root/datasets/$(basename "$DATASET_NAME")"
MODEL_ROOT="/root/models/${MODEL_NAME}"

# Validate MODEL_NAME (Qwen3-VL only, matches the NPU .py orchestrator)
VALID_MODELS="
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$VIME_SCRIPT_EXTERNAL_RAY" ] || [ "$VIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# ---------------------------------------------------------------------------
# NPU environment (from examples/geo3k_vlm_multi_turn/run_grpo_npu.sh wrapper)
# ---------------------------------------------------------------------------
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

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 vime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 vime
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1

# Download model and dataset if missing
mkdir -p /root/models /root/datasets
if [ ! -d "${MODEL_ROOT}" ]; then
   hf download Qwen/${MODEL_NAME} --local-dir ${MODEL_ROOT}
fi
if [ ! -d "${DATA_ROOT}" ]; then
   hf download --repo-type dataset ${DATASET_NAME} --local-dir ${DATA_ROOT}
fi

# Common args
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}
   # qwen3 vl model has rotary base 5000000
   --rotary-base 5000000
)

# Single-turn rollout (no multi-turn custom-generate-function / custom-config)
ROLLOUT_ARGS=(
   --prompt-data ${DATA_ROOT}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 200
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

# Eval disabled for NPU bring-up (minimize moving parts; eval is an extra full
# rollout pass that re-exposes the same NPU rollout bugs). test.parquet exists,
# so re-enable this block + the ${EVAL_ARGS[@]} line below once training is stable.
# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data geo3k_imgurl_processed ${DATA_ROOT}/test.parquet
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 4096
# )

# Note: no reg anchor (kl-coef 0 + entropy-coef 0). The multi-turn NPU run
# diverged ~step128 / NaN-crashed ~step167 this way, but that's well past this
# short bring-up run (num-rollout 200). Add --kl-coef 0.005 for longer runs.
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
   --vllm-gpu-memory-utilization ${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
   --vllm-max-model-len 16384
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project vime-geo3k-vlm
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}-vllm-${NUM_GPUS}npu
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

# megatron backend (NPU-tuned)
BACKEND_ARGS=(
   --train-backend megatron
   --load ${MODEL_ROOT}
   --ref-load ${MODEL_ROOT}
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # P8 workaround: bypass broken THD/varlen attention by using bshd. bshd
   # asserts use_dynamic_batch_size is False -> fixed micro-batch (no
   # --use-dynamic-batch-size/--max-tokens-per-gpu/--balance-data).
   --qkv-format bshd
   --micro-batch-size 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

MISC_ARGS=(
   --no-gradient-accumulation-fusion
   --use-flash-attn
   # Checkpoint planning (disk-aware):
   #   /workspace (=/dev/sda2) is 96% full (~19G) -> cannot hold even one ckpt.
   #   /root/.cache is a ~35T NFS share with ~1.9T free -> save there instead.
   --save /root/.cache/vime-ckpt/geo3k_vlm_single_turn_npu
   # Test run: weights only (~16GB/ckpt) instead of full state (~100GB with
   # optimizer). Drop --no-save-optim if you need to resume the optimizer.
   --no-save-optim
   # No auto-prune of old ckpts. At ~16GB each, keep the interval large and
   # clean /root/.cache/vime-ckpt manually if it grows.
   --save-interval 50
)

# get MODEL_ARGS from scripts/models for megatron backend
VIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
# VL models require rotary-base 5000000
MODEL_ARGS_ROTARY_BASE=5000000 source "${VIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\",
    \"PYTORCH_NPU_ALLOC_CONF\": \"expandable_segments:True\",
    \"VLLM_ASCEND_ENABLE_NZ\": \"0\",
    \"ASCEND_TOOLKIT_HOME\": \"/usr/local/Ascend/ascend-toolkit/latest/\",
    \"ASCEND_OPP_PATH\": \"/usr/local/Ascend/ascend-toolkit/latest/opp/\",
    \"ASCEND_AICPU_PATH\": \"/usr/local/Ascend/ascend-toolkit/latest/\",
    \"ASCEND_HOME_PATH\": \"/usr/local/Ascend/ascend-toolkit/latest/\",
    \"set_env_path\": \"/usr/local/Ascend/nnal/atb/set_env.sh\",
    \"HYDRA_FULL_ERROR\": \"1\",
    \"HCCL_HOST_SOCKET_PORT_RANGE\": \"60000-60050\",
    \"HCCL_NPU_SOCKET_PORT_RANGE\": \"61000-61050\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
  }
}"

LOG_FILE="/root/${MODEL_NAME}-single-turn-${DATE}.log"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]} \
   2>&1 | tee -a "$LOG_FILE"
