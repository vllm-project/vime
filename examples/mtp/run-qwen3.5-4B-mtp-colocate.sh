#!/bin/bash

# MTP speculative decoding for Qwen3.5-4B — colocate mode (8 GPUs)
#
# Qwen3.5 checkpoints ship one multi-token-prediction (MTP) layer. vLLM serves it
# as the speculative draft (method "mtp"); with SPEC_MODE=online the trainer also
# trains the MTP layer during RL and syncs it to vLLM every step. Train and
# rollout share the same GPUs; the policy is offloaded during rollout.
#
# Prerequisites:
#   - Qwen3.5-4B HF checkpoint and two torch_dist conversions next to it:
#       ${MODEL_DIR}_torch_dist      (no MTP layer; SPEC_MODE off and frozen)
#       ${MODEL_DIR}_mtp_torch_dist  (converted with --mtp-num-layers 1; online)
#   - dapo-math-17k dataset
#
# Usage: bash examples/mtp/run-qwen3.5-4B-mtp-colocate.sh
#
# Optional environment:
#   SPEC_MODE=online|frozen|off  online (default) trains the MTP layer during RL
#       and syncs it every step; frozen serves the checkpoint's MTP layer
#       unchanged; off disables speculative decoding. Every mode trains the
#       policy with GRPO.
#   MTP_K (default 2, the Qwen3.5-4B model card's vLLM setting): speculative
#       tokens per step.
#   RM_TYPE (default deepscaler) and CHAT_TEMPLATE_KWARGS (JSON, unset by default):
#       e.g. CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' RM_TYPE=math for
#       short non-thinking answers; deepscaler scores only text after </think>.
#   NUM_GPUS (default 8) and TP_SIZE (default 1): GPUs on this node and
#       training tensor parallelism.
#   NO_SAVE=1 skips checkpoints (including the final one).
#   EVAL_INTERVAL (off disables evaluation, including the one before training),
#   EXTRA_ARGS (appended to train.py), plus the variables below.

set -ex
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/vime/scripts/models/qwen3.5-4B.sh"

MODEL_DIR=${MODEL_DIR:-/root/Qwen3.5-4B}
DATA_PATH=${DATA_PATH:-/root/dapo-math-17k/dapo-math-17k.jsonl}
NUM_GPUS=${NUM_GPUS:-8}
SAVE_DIR=${SAVE_DIR:-/root/Qwen3.5-4B_mtp_colocate/}
SPEC_MODE=${SPEC_MODE:-online}
MTP_K=${MTP_K:-2}

SPEC_ARGS=(
   --vllm-speculative-config '{"method":"mtp","num_speculative_tokens":'"${MTP_K}"'}'
)
MTP_ARGS=()
REF_LOAD="${MODEL_DIR}_torch_dist"
case "${SPEC_MODE}" in
   online)
      MTP_ARGS=(--mtp-num-layers 1 --enable-mtp-training --mtp-loss-scaling-factor 0.2)
      REF_LOAD="${MODEL_DIR}_mtp_torch_dist"
      ;;
   frozen) ;;
   off) SPEC_ARGS=() ;;
   *) echo "SPEC_MODE must be online, frozen or off" >&2; exit 1 ;;
esac

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${REF_LOAD}"
)
if [ "${NO_SAVE:-0}" != "1" ]; then
   CKPT_ARGS+=(--save "${SAVE_DIR}" --save-interval "${SAVE_INTERVAL:-50}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type "${RM_TYPE:-deepscaler}"
   --num-rollout "${NUM_ROLLOUT:-200}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
   --rollout-temperature 1

   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
)

if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-1}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
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
   --rollout-num-gpus-per-engine 1
   --vllm-gpu-memory-utilization 0.5
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL:-100}"
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

if [ "${EVAL_INTERVAL:-}" = "off" ]; then
   EVAL_ARGS=()
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-persist-layer-norm
   --loss-mask-type qwen3_5
)

# Start Ray with all GPUs
ray start --head --port=6379 --num-gpus="${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/vime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"${TENSORBOARD_DIR:+,
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\"}
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /root/vime \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${SPEC_ARGS[@]} \
   ${MTP_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${EXTRA_ARGS:-}

# Cleanup
ray stop --force
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py" 2>/dev/null || true
