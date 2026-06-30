#!/usr/bin/env bash
set -o pipefail

export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTHONPATH="/workspace/issue205/Megatron-Bridge/src:/workspace/issue205/Megatron-Bridge:/workspace/issue205/Megatron-LM:/workspace/issue205/vime-ascend:${PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0

VIME_ROOT="${VIME_ROOT:-/workspace/issue205/vime-ascend}"
SCRIPT_DIR="${VIME_ROOT}/scripts"
WEIGHT_DIR="${WEIGHT_DIR:-/home/data/weights/Qwen3-4B}"
DATA_FILE="${DATA_FILE:-/home/w00893744/dataset/dapo-math-17k.jsonl}"
LOG_FILE="${LOG_FILE:-/home/w00893744/train_qwen3_4b_multi_agent_vllm.log}"

source "${SCRIPT_DIR}/models/qwen3-4B.sh"

python "${VIME_ROOT}/train.py" \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${WEIGHT_DIR}" \
  --custom-generate-function-path examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents \
  --prompt-data "${DATA_FILE}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type math \
  --rollout-backend vllm \
  --vllm-weight-sync-mode native \
  --vllm-gpu-memory-utilization 0.6 \
  --vllm-enable-sleep-mode \
  --vllm-max-model-len 4096 \
  --vllm-enforce-eager \
  --num-rollout 200 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-context-len 4096 \
  --rollout-max-response-len 2048 \
  --rollout-temperature 1.0 \
  --global-batch-size 256 \
  --balance-data \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.00 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --load "${WEIGHT_DIR}" \
  --megatron-to-hf-mode bridge \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --micro-batch-size 1 \
  --use-flash-attn \
  --train-memory-margin-bytes 2147483648 \
  2>&1 | tee -a "${LOG_FILE}"
