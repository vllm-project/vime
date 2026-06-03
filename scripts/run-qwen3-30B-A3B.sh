#!/bin/bash

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TOTAL_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))

if [[ "${ACTOR_NUM_NODES}" -le 1 ]]; then
   # for rerun the task (single-node only)
   pkill -9 -f "vllm serve"
   sleep 3
   ray stop --force
   pkill -9 ray
   pkill -9 python
   sleep 3
   pkill -9 ray
   pkill -9 python
   pkill -9 redis
else
   export SLIME_SCRIPT_EXTERNAL_RAY=1
   export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to head LAN IP}"
fi

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

if [[ "${ACTOR_NUM_NODES}" -le 1 ]]; then
   CKPT_ARGS=(
      --hf-checkpoint /root/Qwen3-30B-A3B
      #--hf-checkpoint /root/Qwen3-30B-A3B-FP8
      --ref-load /root/Qwen3-30B-A3B_torch_dist
      --load /root/Qwen3-30B-A3B_slime/
      --save /root/Qwen3-30B-A3B_slime/
      --save-interval 20
   )

   ROLLOUT_ARGS=(
      --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
      --input-key prompt
      --label-key label
      --apply-chat-template
      --rollout-shuffle
      --rm-type deepscaler
      --num-rollout 3000
      --rollout-batch-size 32
      --n-samples-per-prompt 8
      --rollout-max-response-len 8192
      --rollout-temperature 1

      --global-batch-size 256
      --balance-data
   )

   EVAL_ARGS=(
      --eval-interval 20
      --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
      --n-samples-per-eval-prompt 16
      --eval-max-response-len 16384
      --eval-top-p 1
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

      # --micro-batch-size 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu 20480
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

      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )

   WANDB_ARGS=(
      #--use-wandb
      # --wandb-project vime-dev
      # --wandb-group qwen3-30B-A3B-test
      # --wandb-key ${WANDB_KEY}
   )

   VLLM_ARGS=(
      --rollout-num-gpus-per-engine 8
      --vllm-gpu-memory-utilization 0.7
      --vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256)
   )

   MISC_ARGS=(
      # default dropout in megatron is 0.1
      --attention-dropout 0.0
      --hidden-dropout 0.0
      # should be good for model performance
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      # need to comment this when using model with MLA
      --attention-backend flash
   )

else
   MEGATRON_TP="${MEGATRON_TP:-8}"
   MEGATRON_PP="${MEGATRON_PP:-1}"
   MEGATRON_CP="${MEGATRON_CP:-2}"
   MEGATRON_EP="${MEGATRON_EP:-8}"
   ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-${TOTAL_GPUS}}"
   NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${ACTOR_NUM_GPUS_PER_NODE}}"
   NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
   ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
   N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
   ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
   GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
   ENABLE_R3="${ENABLE_R3:-1}"
   HF_CKPT="${HF_CKPT:-/root/Qwen3-30B-A3B}"
   TORCH_DIST="${TORCH_DIST:-/root/models/Qwen3-30B-A3B_torch_dist}"
   PROMPT_DATA="${PROMPT_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}"

   CKPT_ARGS=(
      --hf-checkpoint "${HF_CKPT}"
      --ref-load "${TORCH_DIST}"
   )

   ROLLOUT_ARGS=(
      --prompt-data "${PROMPT_DATA}"
      --input-key prompt
      --label-key label
      --apply-chat-template
      --rollout-shuffle
      --rm-type deepscaler
      --num-rollout "${NUM_ROLLOUT}"
      --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
      --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
      --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
      --rollout-temperature 1
      --global-batch-size "${GLOBAL_BATCH_SIZE}"
      --balance-data
   )

   EVAL_ARGS=()
   WANDB_ARGS=()

   PERF_ARGS=(
      --tensor-model-parallel-size "${MEGATRON_TP}"
      --sequence-parallel
      --pipeline-model-parallel-size "${MEGATRON_PP}"
      --context-parallel-size "${MEGATRON_CP}"
      --expert-model-parallel-size "${MEGATRON_EP}"
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
   )

   GRPO_ARGS=(
      --advantage-estimator gspo
      --kl-loss-coef 0.00
      --kl-loss-type low_var_kl
      --kl-coef 0.00
      --entropy-coef 0.00
      --eps-clip 4e-4
   )
   if [[ "${ENABLE_R3}" == "1" ]]; then
      GRPO_ARGS+=(--use-rollout-routing-replay)
   fi

   OPTIMIZER_ARGS=(
      --optimizer adam
      --lr 1e-6
      --lr-decay-style constant
      --weight-decay 0.1
      --adam-beta1 0.9
      --adam-beta2 0.98
   )

   VLLM_ARGS=(
      --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
      --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
      --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
      --vllm-moe-backend "${VLLM_MOE_BACKEND:-triton}"
   )
   if [[ "${ENABLE_R3}" == "1" ]]; then
      VLLM_ARGS+=(
         --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-6144}"
         --vllm-server-concurrency "${VLLM_SERVER_CONCURRENCY:-96}"
         --update-weight-buffer-size "${UPDATE_WEIGHT_BUFFER_SIZE:-$((512 * 1024 * 1024))}"
      )
   fi

   MISC_ARGS=(
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash
      --moe-token-dispatcher-type alltoall
   )
fi

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

if [[ "${ACTOR_NUM_NODES}" -le 1 ]]; then
   # launch the master node of ray in container
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

   RUNTIME_ENV_JSON="{
     \"env_vars\": {
       \"PYTHONPATH\": \"/root/Megatron-LM/\",
       \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
       \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
     }
   }"
else
   RUNTIME_ENV_JSON="{
     \"env_vars\": {
       \"PYTHONPATH\": \"${VIME_ROOT}:/root/Megatron-LM/\",
       \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
       \"MASTER_ADDR\": \"${MASTER_ADDR}\",
       \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-ens90f0np0,ens90f1np1,ens92f0np0,ens92f1np1}\",
       \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-enp130s0f0}\",
       \"NCCL_IB_DISABLE\": \"${NCCL_IB_DISABLE:-1}\",
       \"SLIME_NCCL_BRIDGE_CPU_FALLBACK\": \"${SLIME_NCCL_BRIDGE_CPU_FALLBACK:-1}\",
       \"VLLM_SERVER_DEV_MODE\": \"1\"
     }
   }"
   cd "${VIME_ROOT}"
fi

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
