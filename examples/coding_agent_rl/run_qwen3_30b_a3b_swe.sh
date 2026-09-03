#!/usr/bin/env bash
# End-to-end SWE coding-agent RL on single a3 node.
#
# Run from a long-lived shell / tmux session on the Ray head node; do not wrap
# in a short-lived nohup launcher or Ray child processes get cleaned up with it.

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::' || true
pkill -9 -f VLLM || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
pkill -9 redis || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_DIR="${VIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${VIME_DIR}/scripts/models/qwen3-30B-A3B.sh"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-40960}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/home/vllm/weights/Qwen3-30B-A3B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/home/vllm/weights/Qwen3-30B-A3B_torch_dist_8cards}"
PROMPT_DATA="${PROMPT_DATA:-/home/vllm/c00944022/datasets/swebench_verified/swe_train.jsonl}"

EXP_TAG="${EXP_TAG:-agent_only}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${VIME_DIR}/runs/${EXP_TAG}_${STAMP}}"

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps"
LOG_FILE="${LOG_DIR}/run.log"
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "======================================================================"


CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --load "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   # --debug-rollout-only
   # --debug-train-only
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --rollout-stop-token-ids 151645 151643
   --num-steps-per-rollout 1
   --global-batch-size 64
   --micro-batch-size 1
   # --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
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
   # max-tokens-per-gpu is one CP rank's slice of MAX_CONTEXT_LEN; log-probs are
   # chunked along T to avoid OOM on long single trajectories.
   --max-tokens-per-gpu 40960
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-4
   --eps-clip-high 2e-4
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
   --rollout-num-gpus-per-engine 8
   --vllm-gpu-memory-utilization 0.75
   --vllm-tool-call-parser qwen3_coder
   --vllm-reasoning-parser qwen3
   # --prefill-num-servers 1
   # --vllm-enforce-eager
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # --moe-token-dispatcher-type flex
   # --moe-enable-deepep
   --use-flash-attn
   --no-gradient-accumulation-fusion
)

# Set MASTER_ADDR before the SWE block
export MASTER_ADDR="192.168.13.190"
export VIME_HEAD_HOST="${MASTER_ADDR}"

# ============ SWE / claude-code rollout knobs ============
E2B_DUMMY_API_KEY="e2b_0000000000000000000000000000000000000000"
SANDBOX_METADATA_FILE=/dev/null
export E2B_API_KEY="${E2B_DUMMY_API_KEY}"
export SWE_SANDBOX_METADATA_FILE="${SANDBOX_METADATA_FILE}"

export DOCKER_SANDBOX=1
export DOCKER_SANDBOX_HOST="root@192.168.13.188"
export DOCKER_TARBALL_DIR="/home/vllm/c00944022/vime-agent/env"
export SWE_HOST_NODE_TARBALL="${DOCKER_TARBALL_DIR}/node-v24.14.0-linux-x64.tar.xz"
export SWE_HOST_CC_TARBALL="${DOCKER_TARBALL_DIR}/anthropic-ai-claude-code-2.1.226.tgz"
export SWE_HOST_CC_TARBALL_DEP="${DOCKER_TARBALL_DIR}/claude-code-linux-x64-2.1.226.tgz"

# --- per-trajectory time / concurrency budgets ---
export SWE_TIME_BUDGET_SEC="${SWE_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-6}"

# --- claude-code CLI extras ---
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":80000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SWE_CLAUDE_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${VIME_HEAD_HOST}"
export NO_PROXY="${no_proxy}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ============ Ascend env vars ============
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="/home/vllm/c00944022/vime-proj/Megatron-Bridge/src:/home/vllm/c00944022/vime-proj/Megatron-LM/:${PYTHONPATH:-}"


ray start --head --node-ip-address "${MASTER_ADDR}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8267 --dashboard-agent-listen-port=52367

echo "Waiting for Ray cluster to stabilize..."
sleep 30
ray status

ray job submit --address="http://${MASTER_ADDR}:8267" \
   -- python3 -u train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${VLLM_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
