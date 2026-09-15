#!/bin/bash

# DSpark speculative decoding training — non-colocate mode (8 GPUs)
#
# Train and rollout run on separate GPU groups. The policy model stays on GPU
# during rollout (no offload overhead), and draft weights are synced to vLLM
# via packed tensor transfer.
#
# GPU layout:
#   GPUs 0-3: Policy Megatron training (TP=4)
#   GPUs 4-7: vLLM rollout with DSpark speculative decoding
#
# Prerequisites:
#   - Qwen3-4B HF checkpoint + torch_dist conversion
#   - Pre-trained DSpark draft checkpoint (model.safetensors)
#   - dapo-math-17k dataset
#
# Usage: bash examples/dspark/run-qwen3-4B-dspark-non-colocate.sh

set -ex
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/vime/scripts/models/qwen3-4B.sh"

MODEL_DIR=${MODEL_DIR:-/root/Qwen3-4B}
DSPARK_MODEL=${DSPARK_MODEL:-/root/Qwen3-4B-dspark-pretrained}
DATA_PATH=${DATA_PATH:-/root/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-/root/Qwen3-4B_dspark_non_colocate/}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-50}"
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT:-200}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
   --rollout-temperature 1

   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
)

PERF_ARGS=(
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
   --rollout-num-gpus 4
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.85
   --vllm-speculative-config '{"method":"dspark","model":"'"${DSPARK_MODEL}"'","num_speculative_tokens":7}'
)

DSPARK_ARGS=(
   --dspark-block-size 7
   --dspark-num-draft-layers 5
   --dspark-target-layer-ids 1,9,17,25,33
   --dspark-markov-rank 256
   --dspark-markov-head-type vanilla
   --dspark-num-anchors 512
   --dspark-mask-token-id 151669
   --dspark-ce-loss-alpha 0.5
   --dspark-l1-loss-alpha 0.5
   --dspark-confidence-head-alpha 0.1
   --dspark-loss-decay-gamma 4.0
   --dspark-draft-loss-weight 1.0
   --dspark-freeze-policy
   --dspark-intermediate-size 9728
   --dspark-pretrained-model "${DSPARK_MODEL}/model.safetensors"
)

EVAL_ARGS=(
   --eval-interval 100
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-persist-layer-norm
)

# Start Ray with all 8 GPUs
ray start --head --port=6379 --num-gpus=8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/vime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir /root/vime \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${DSPARK_ARGS[@]} \
   ${MISC_ARGS[@]}

# Cleanup
ray stop --force
pkill -9 ray 2>/dev/null || true
pkill -9 -f "train.py" 2>/dev/null || true
