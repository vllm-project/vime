#!/bin/bash

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
pkill -9 -f VLLM
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="/root/Megatron-LM/:/root/verl-SpeCo:${PYTHONPATH:-}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

TARGET_MODEL="${TARGET_MODEL:-/data/weights/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-/data/weights/Qwen3-4B-eagle3-angelslim}"
PROMPT_DATA="${PROMPT_DATA:-/data/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
DRAFT_CHECKPOINT_PATH="${DRAFT_CHECKPOINT_PATH:-/data/eagle3-output/draft}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8665}"
RAY_GCS_PORT="${RAY_GCS_PORT:-8679}"
RAY_AGENT_LISTEN_PORT="${RAY_AGENT_LISTEN_PORT:-52377}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_vime_new_eagle3_8c}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1}"

CKPT_ARGS=(
   --hf-checkpoint "${TARGET_MODEL}/"
   --load "${TARGET_MODEL}/"
   --ref-load "${TARGET_MODEL}/"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.0
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

VLLM_ARGS=(
   --vllm-additional-config '{"weight_nz_mode":0}'
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.6
   --vllm-max-model-len 4096
   --vllm-speculative-config "{\"method\":\"eagle3\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":3}"
)

DRAFT_ARGS=(
   --enable-external-draft-training
   --draft-algorithm eagle3
   --draft-model-path "${DRAFT_MODEL}"
   --draft-model-factory-path vime.backends.speculative_training.factories.verl_speco_eagle3.build_model
   --draft-feature-layer-ids 1,17,32
   --draft-checkpoint-path "${DRAFT_CHECKPOINT_PATH}"
   --draft-save-interval 1
   --update-weight-mode full
   --update-weight-transport nccl
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --micro-batch-size 1
   --use-flash-attn
)

mkdir -p "$(dirname "${DRAFT_CHECKPOINT_PATH}")"

ray start --head --num-gpus 0 --resources '{"NPU": 8}' --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-agent-listen-port="${RAY_AGENT_LISTEN_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}" --port="${RAY_GCS_PORT}" --temp-dir="${RAY_TEMP_DIR}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
-- python3 train.py \
--actor-num-nodes 1 \
--actor-num-gpus-per-node 4 \
--rollout-num-gpus 4 \
${MODEL_ARGS[@]} \
${CKPT_ARGS[@]} \
${ROLLOUT_ARGS[@]} \
${OPTIMIZER_ARGS[@]} \
${GRPO_ARGS[@]} \
${PERF_ARGS[@]} \
${VLLM_ARGS[@]} \
${DRAFT_ARGS[@]} \
${MISC_ARGS[@]}
