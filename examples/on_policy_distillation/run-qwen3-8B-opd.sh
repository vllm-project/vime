#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -ex

export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Start the teacher model server
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
LOG_FILE="/tmp/vllm_teacher_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

## Launch the teacher model server in the background
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
    --model /root/Qwen3-32B \
    --host 0.0.0.0 \
    --port ${TEACHER_PORT} \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --disable-custom-all-reduce \
    > "${LOG_FILE}" 2>&1 &
TEACHER_PID=$!

echo "Starting teacher model server (pid=${TEACHER_PID})..."

## Wait for the teacher model server to be ready
for i in $(seq 1 120); do
    if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
        echo "ERROR: Teacher server process died. Check ${LOG_FILE}"
        tail -n 20 "${LOG_FILE}"
        exit 1
    fi
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${TEACHER_IP}:${TEACHER_PORT}/health" 2>/dev/null || true)
    if [ "${HTTP_CODE}" = "200" ]; then
        echo "Teacher model server is up and running at ${TEACHER_IP}:${TEACHER_PORT}."
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Teacher server failed to start within 10 minutes"
        tail -n 20 "${LOG_FILE}"
        kill "${TEACHER_PID}" 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for the teacher model server to start..."
    sleep 5
done
sleep 5

source "/root/vime/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_torch_dist
   --save /root/Qwen3-8B_vime/
   --save-interval 20
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-max-context-len 8192
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

RM_ARGS=(
   --custom-rm-path vime.rollout.on_policy_distillation.reward_func
   --custom-reward-post-process-path vime.rollout.on_policy_distillation.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/inference/v1/generate
)

EVAL_ARGS=(
   # --eval-interval 50
   # --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   # --eval-input-key messages
   # --n-samples-per-eval-prompt 1
   # --eval-max-response-len 4096
   # --eval-top-k 1
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
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type vllm
   --opd-kl-coef 1.0
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
   #--use-wandb
   # --wandb-project vime-opd
   # --wandb-group qwen3-8B-opd
   # --wandb-key ${WANDB_KEY}
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1
   --vllm-gpu-memory-utilization 0.25
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --make-vocab-size-divisible-by 128
)

# launch the master node of ray in container
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

#### clear after training
kill ${TEACHER_PID} 2>/dev/null || true
sleep 3
ray stop --force
pkill -9 ray
pkill -9 -f "train.py" || true
sleep 3
