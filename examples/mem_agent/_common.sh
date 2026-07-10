#!/usr/bin/env bash
# Shared setup for MemAgent example scripts (sourced, not executed directly).
set -euo pipefail

MEM_AGENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${MEM_AGENT_DIR}/../.." &>/dev/null && pwd)"

export VIME_ROOT
export PYTHONBUFFERED="${PYTHONBUFFERED:-16}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR},localhost"
export NO_PROXY="${no_proxy}"

# Override via env for your cluster (H800 example in README).
export HF_CKPT="${HF_CKPT:-/data/models/Qwen3-4B}"
export TORCH_DIST="${TORCH_DIST:-/data/models/Qwen3-4B_torch_dist}"
export TRAIN_DATA="${TRAIN_DATA:-/data/datasets/hotpotqa_slime/train.jsonl}"
export SAVE_PATH="${SAVE_PATH:-/data/models/MemAgent_Qwen3-4B-RL}"
export DATA_ROOT="${DATA_ROOT:-/data/datasets/hotpotqa_hf}"
export ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-${HF_CKPT}}"

export MEM_CHUNK_TOKENS="${MEM_CHUNK_TOKENS:-2048}"
export MEM_MAX_MEMORY="${MEM_MAX_MEMORY:-1024}"
export MEM_MAX_FINAL="${MEM_MAX_FINAL:-256}"
export MEM_MAX_CHUNKS="${MEM_MAX_CHUNKS:-64}"

mem_agent_detect_nvlink() {
  local count
  count=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
  export NCCL_NVLS_ENABLE=$([[ "${count}" -gt 0 ]] && echo 1 || echo 0)
}

mem_agent_detect_gpus() {
  local detected=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
  export NUM_GPUS="${NUM_GPUS:-${detected:-8}}"
  if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -le 0 ]]; then
    export NUM_GPUS=8
  fi
}

mem_agent_cleanup() {
  pkill -9 -f '[v]llm serve|VLLM::' 2>/dev/null || true
  sleep 2
  ray stop --force 2>/dev/null || true
  pkill -9 -f '[r]ay::' 2>/dev/null || true
  pkill -9 -f '[t]rain.py' 2>/dev/null || true
  pkill -9 redis 2>/dev/null || true
  rm -rf /tmp/ray /tmp/ray_session_* "${HOME}/.ray" 2>/dev/null || true
  sleep 2
}

mem_agent_rollout_args() {
  ROLLOUT_ARGS=(
    --custom-generate-function-path examples.mem_agent.rollout.generate
    --custom-rm-path examples.mem_agent.rollout.reward_func
    --custom-convert-samples-to-train-data-path examples.mem_agent.custom_convert.custom_convert
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --rollout-shuffle
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 8192
    --rollout-max-context-len 32768
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    --rollout-function-path vime.rollout.vllm_rollout.generate_rollout
  )
}

mem_agent_train_args() {
  CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${TORCH_DIST}"
  )
  if [[ -n "${SAVE_PATH:-}" ]]; then
    CKPT_ARGS+=(--save "${SAVE_PATH}")
    if [[ -n "${SAVE_INTERVAL:-}" ]]; then
      CKPT_ARGS+=(--save-interval "${SAVE_INTERVAL}")
    fi
  fi

  EVAL_ARGS=(--skip-eval-before-train)

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
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --eps-clip 0.2
    --eps-clip-high 0.3
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
    --vllm-gpu-memory-utilization 0.7
    --vllm-max-model-len 32768
    --router-policy consistent_hash
  )

  MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --train-memory-margin-bytes 2147483648
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${NUM_GPUS}"
    --colocate
  )
}

mem_agent_launch_train() {
  cd "${VIME_ROOT}"
  source "${VIME_ROOT}/scripts/models/qwen3-4B.sh"
  mem_agent_rollout_args
  mem_agent_train_args

  export RAY_DISABLE_DOCKER_CPU_WARNING=1
  export PYTHONPATH="/root/Megatron-LM:${VIME_ROOT}:${PYTHONPATH:-}"

  if [[ "${RUN_TRAIN_DIRECT:-0}" == "1" && -n "${RUN_TRAIN_DIRECT_PY:-}" && -f "${RUN_TRAIN_DIRECT_PY}" ]]; then
    export VIME_ROOT NCCL_NVLS_ENABLE MEM_CHUNK_TOKENS MEM_MAX_MEMORY MEM_MAX_FINAL MEM_MAX_CHUNKS
    python3 "${RUN_TRAIN_DIRECT_PY}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${VLLM_ARGS[@]}" \
      "${MISC_ARGS[@]}"
    return
  fi

  ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

  local runtime_env
  runtime_env="{
    \"env_vars\": {
      \"PYTHONPATH\": \"${VIME_ROOT}:/root/Megatron-LM/\",
      \"VIME_ROOT\": \"${VIME_ROOT}\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
      \"PYTHONBUFFERED\": \"16\",
      \"MASTER_ADDR\": \"${MASTER_ADDR}\",
      \"no_proxy\": \"${no_proxy}\",
      \"NO_PROXY\": \"${NO_PROXY}\",
      \"MEM_CHUNK_TOKENS\": \"${MEM_CHUNK_TOKENS}\",
      \"MEM_MAX_MEMORY\": \"${MEM_MAX_MEMORY}\",
      \"MEM_MAX_FINAL\": \"${MEM_MAX_FINAL}\",
      \"MEM_MAX_CHUNKS\": \"${MEM_MAX_CHUNKS}\"
    }
  }"

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${runtime_env}" \
    -- python3 train.py \
      --train-backend megatron \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${GRPO_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${VLLM_ARGS[@]}" \
      "${MISC_ARGS[@]}"
}
