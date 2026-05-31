#!/bin/bash

if grep -q $'\r' "$0" 2>/dev/null; then
  exec bash <(sed 's/\r$//' "$0") "$@"
fi

pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'python3 train.py' 2>/dev/null || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1
export PYTHONBUFFERED=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_ALLOC_CONF 2>/dev/null || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
TOTAL_GPUS="$((ACTOR_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))"
echo "TOTAL_GPUS (ray): ${TOTAL_GPUS}  (actor=${ACTOR_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS}, per_engine=${ROLLOUT_GPUS_PER_ENGINE})"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
VIME_ROOT="${VIME_ROOT:-${REPO_ROOT}}"
cd "${VIME_ROOT}"

source "${VIME_ROOT}/scripts/models/qwen3-4B.sh"

HF_CKPT="${HF_CKPT:-/data/nfs_87/xky/models/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-/data/nfs_87/xky/models/Qwen3-4B_torch_dist}"
SAVE_DIR="${SAVE_DIR:-/data/nfs_87/xky/RL/vime_checkpoints/Qwen3-4B_tau_bench}"

LOG_ROOT="${LOG_ROOT:-/data/nfs_87/xky/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_ROOT}/tb_qwen3_4b_tau_bench_${TS}}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/train_qwen3_4b_tau_bench_${TS}.log}"
mkdir -p "${TENSORBOARD_DIR}" "${LOG_ROOT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
)

PROMPT_DATA="${PROMPT_DATA:-/data/nfs_87/xky/datasets/tau-bench/retail_train_tasks.jsonl}"
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data retail-dev /data/nfs_87/xky/datasets/tau-bench/retail_dev_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
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
   # --use-wandb
   # --wandb-project vime-tau-bench
   # --wandb-group qwen3-4B
   # --wandb-key ${WANDB_KEY}
)

VLLM_ARGS=(
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --rollout-backend vllm
   --vllm-weight-sync-mode native
   --vllm-gpu-memory-utilization 0.7
)
export SLIME_VLLM_SERVER_HEALTH_TIMEOUT_SEC="${SLIME_VLLM_SERVER_HEALTH_TIMEOUT_SEC:-900}"

CUSTOM_ARGS=(
   --custom-generate-function-path examples.tau_bench.rollout.generate
   --custom-config-path examples/tau_bench/tau_bench_config.yaml
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --train-memory-margin-bytes 2147483648
   --use-tensorboard
)

VIME_PYTHONPATH="${VIME_ROOT}:/root/Megatron-LM/"

echo "VIME_ROOT=${VIME_ROOT}"
echo "HF_CKPT=${HF_CKPT} REF_LOAD=${REF_LOAD} SAVE=${SAVE_DIR}"
echo "PROMPT_DATA=${PROMPT_DATA}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${TOTAL_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${VIME_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"SLIME_VLLM_SERVER_HEALTH_TIMEOUT_SEC\": \"${SLIME_VLLM_SERVER_HEALTH_TIMEOUT_SEC}\",
    \"GEMINI_API_KEY\": \"${GEMINI_API_KEY:-NONE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   2>&1 | tee -a "${LOG_FILE}"
