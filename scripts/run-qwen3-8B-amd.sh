#!/bin/bash
# Colocate GRPO for Qwen3-8B on ROCm (gfx950 / MI350). Short validation run by
# default; override the knobs below to scale up.

# Clean leftovers from a previous run (vLLM orphans procs named VLLM::*).
ray stop --force
pkill -9 -f "VLLM::"
pkill -9 -f "EngineCore"
pkill -9 -f "ray::"
pkill -9 -f "raylet|gcs_server|ray/dashboard|default_worker|log_monitor|runtime_env_agent|autoscaler"
pkill -9 -f "train.py"
sleep 3
pkill -9 -f "VLLM::"

set -ex

export PYTHONUNBUFFERED=1

# Clear baked NVTE_* so Megatron sets the attn backend from --attention-backend flash.
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN

# ----- ROCm GPU visibility: single TP=2 engine; keep HIP == CUDA visibility -----
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-1}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}
export HIP_VISIBLE_DEVICES=${VISIBLE_GPUS:-${HIP_VISIBLE_DEVICES:-6,7}}
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"

IFS=',' read -r -a _visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${NUM_GPUS:-${#_visible_gpu_ids[@]}}
HAS_NVLINK=0                      # AMD: no NVLink; disable NCCL NVLS
echo "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES  NUM_GPUS=$NUM_GPUS  HAS_NVLINK=$HAS_NVLINK"

# ----- short-run knobs (override to scale up to a real job) -----
NUM_ROLLOUT=${NUM_ROLLOUT:-3}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}   # effectively off for the short run

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_vime/
   --save /root/Qwen3-8B_vime/
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
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${MAX_RESPONSE_LEN}
   --rollout-temperature 1

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# Eval disabled (no --eval-prompt-data). Add it back to measure accuracy.
EVAL_ARGS=()

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
   --max-tokens-per-gpu 4096
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

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT:-vime-qwen3-8B-rocm}"
   --wandb-group "${WANDB_GROUP:-qwen3-8B-grpo-rocm}"
   --wandb-key "${WANDB_API_KEY:-${wandb_key}}"
   --wandb-mode online
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 2
   # Modest KV reservation so Megatron's post-rollout onload doesn't OOM at TP=2.
   --vllm-gpu-memory-utilization 0.4
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --train-memory-margin-bytes 2147483648
   # APEX not installed on this ROCm image -> disable fused grad accumulation.
   --no-gradient-accumulation-fusion
   # Keep train state resident: the colocate offload path leaks VRAM on ROCm gfx950
   # (torch_memory_saver VMM) and Qwen3-8B fits alongside vLLM, so offload is unneeded.
   --no-offload-train
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
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
