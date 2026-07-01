#!/bin/bash

# GLM-5.2 744B-A40B RL training on 18 GB300 trays / 72 GPUs.
# Prerequisite: Ray cluster must be running (use setup-ray-cluster.sh).

set -ex

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm5.2-744B-A40B.sh"

MASTER_ADDR="${MASTER_ADDR:-10.13.84.13}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/weka/models}"
DATA_ROOT="${DATA_ROOT:-/mnt/weka/aoshen/data/dapo-math-17k-hf}"
RUN_ID="$(date +%Y%m%d-%H%M%S)-$(cat /proc/sys/kernel/random/uuid | cut -d- -f1)"
LOG_DIR="${PROJECT_ROOT}/agent_run/results/glm52-training-run-$(date +%Y%m%d)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train-${RUN_ID}.log"

SOCKET_IFNAME=${SOCKET_IFNAME:-bond0.225}

CKPT_ARGS=(
   --hf-checkpoint $MODEL_ROOT/GLM-5.2-FP8
   --ref-load $MODEL_ROOT/GLM-5.2_torch_dist
   --load $MODEL_ROOT/GLM-5.2_vime
   --save $MODEL_ROOT/GLM-5.2_vime
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $DATA_ROOT/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 65536
   --rollout-temperature 1.0

   --global-batch-size 64
)

# TP=8, PP=4, CP=2 consumes all 64 training GPUs (16 trays x 4 GPUs; DP=1).
# Experts use EP=16: expert_tp(1) * ep(16) * pp(4) = 64 = world size (expert_dp=1).
#
# DSA cross-layer index sharing requires every pipeline stage to START on a
# "computing" layer. With PP=4: first=20, mid=20, last=18.
# Stage starts land on global layers 0,20,40,60 -- all computing layers.
PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --decoder-first-pipeline-num-layers 18
   --decoder-last-pipeline-num-layers 20
   --context-parallel-size 2
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 65536
   --data-pad-size-multiplier 1024
   --log-probs-chunk-size 65536
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
   --tis-clip-low 0.5
   --tis-clip 2.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project vime-dev
   # --wandb-group glm5.2-744B-A40B
)

VLLM_CONFIG_FILE="${SCRIPT_DIR}/vllm_glm52_744B_A40B.yaml"

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 8
   --vllm-gpu-memory-utilization 0.80
   --vllm-max-model-len 131072
   --vllm-enable-expert-parallel
   --vllm-enable-ep-weight-filter
   --vllm-speculative-config '{"method":"mtp","num_speculative_tokens":4}'
   --vllm-config "${VLLM_CONFIG_FILE}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --moe-token-dispatcher-type alltoall
)

NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},10.0.0.0/8,100.64.0.0/10"
export no_proxy="${NO_PROXY_LIST}"
export NO_PROXY="${NO_PROXY_LIST}"

echo "Logging to ${LOG_FILE}"

ray job submit --address="http://${MASTER_ADDR}:8265" \
   --working-dir "${PROJECT_ROOT}" \
   --runtime-env-json="$(cat <<EOF_JSON
{
  "excludes": ["reference/", "agent_run/results/", ".git/"],
  "env_vars": {
    "PYTHONPATH": "${PROJECT_ROOT}:/root/Megatron-LM/",
    "PYTHONUNBUFFERED": "1",
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "3600",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600"
  }
}
EOF_JSON
)" \
   -- python3 train.py \
   --actor-num-nodes 16 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   --no-check-for-nan-in-loss-and-grad \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 2 )) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
