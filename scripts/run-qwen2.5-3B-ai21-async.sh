#!/bin/bash
#
# Fully-async variant of scripts/run-qwen2.5-3B-ai21-regression-v2.sh
# (Qwen2.5-3B-Instruct GRPO, AI21 features, asynchronous training).
#
# Deltas vs the synchronous regression script — everything else is identical:
#   1. Driver: train_async.py (overlaps generation of step N+1 with training
#      of step N) instead of train.py.
#   2. No --colocate: train and rollout live on SEPARATE GPUs
#      (ACTOR_GPUS + ROLLOUT_GPUS = NUM_GPUS; default 50/50 split).
#   3. Rollout function: vime_plugins.rollout.fully_async_ai21 — vime's
#      fully-async worker (continuous in-flight generation pool across step
#      boundaries) with the AI21 seams re-attached (snoozing curriculum,
#      filtered-out dump, prefilter metrics, drop counters). The stock
#      vime.rollout.fully_async_rollout path BYPASSES all of those.
#   4. --eval-function-path pinned to the stock synchronous generate_rollout
#      (it defaults to the rollout function path, which raises on eval).
#
# NOT a faithful regression of the verl run: fully-async training is
# off-policy (in-flight generations span weight updates). Use the v2 script
# for parity checks; use this one for throughput.
#
# Run:  bash scripts/run-qwen2.5-3B-ai21-async.sh

set -ex

# ---- rerun cleanup (mirrors the other run-*.sh scripts) ----
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
# vLLM renames its workers (VLLM::EngineCore / APIServer), so pkill-by-name misses
# them and orphans from a crashed run keep ~85% of every GPU allocated.
pkill -9 -f 'VLLM' || true
pkill -9 -f 'vllm' || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
# Last resort: kill anything still holding GPU memory.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 || true
fi
sleep 2

export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Cluster sizing — async needs disjoint train/rollout GPU sets.
# 3B is TP=1/PP=1 (pure data-parallel), so any split works; default 50/50.
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
[ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ] && NUM_GPUS=4
ACTOR_GPUS=${ACTOR_GPUS:-$((NUM_GPUS / 2))}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}
echo "NUM_GPUS: $NUM_GPUS (train: $ACTOR_GPUS, rollout: $ROLLOUT_GPUS)"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-3B.sh"   # provides MODEL_ARGS (Qwen2.5-3B arch)

# ---------------------------------------------------------------------------
# Paths (same as the regression script)
# ---------------------------------------------------------------------------
HF_CKPT=${HF_CKPT:-/root/Qwen2.5-3B-Instruct}
REF_LOAD=${REF_LOAD:-/root/Qwen2.5-3B-Instruct_torch_dist}
VIME_CKPT=${VIME_CKPT:-/root/Qwen2.5-3B-Instruct_vime_async}   # separate dir: don't clobber the regression ckpts

TRAIN_DATA=${TRAIN_DATA:-/root/regression_short_train.jsonl}   # see DATA PREP in the v2 script
EVAL_DATA=${EVAL_DATA:-/root/regression_short_test.jsonl}

# ---------------------------------------------------------------------------
# AI21 plugin config (identical to the regression script)
# ---------------------------------------------------------------------------
export AI21_EVALUATORS_TIMEOUT=${AI21_EVALUATORS_TIMEOUT:-120.0}
export AI21_CLEAN_THINKING_TRACE=${AI21_CLEAN_THINKING_TRACE:-true}
export AI21_SNOOZE_NUM_TIMES=${AI21_SNOOZE_NUM_TIMES:-5}
export AI21_SNOOZE_MEAN_SCORE_THRESHOLD=${AI21_SNOOZE_MEAN_SCORE_THRESHOLD:-0.99}
export AI21_SNOOZE_ID_KEY=${AI21_SNOOZE_ID_KEY:-id}
export AI21_FILTERED_ROLLOUT_DUMP_PATH=${AI21_FILTERED_ROLLOUT_DUMP_PATH:-/dev/shm/generations/rollout_filtered_out.jsonl}
export AI21_CONFIG_DUMP_PATH=${AI21_CONFIG_DUMP_PATH:-auto}

# ---------------------------------------------------------------------------
# Batch sizing — same formula as the regression script (scaled by TOTAL GPUs
# so step sizes stay comparable at the same NUM_GPUS).
# ---------------------------------------------------------------------------
ROLLOUT_BATCH_SIZE=$((8 * NUM_GPUS))            # 32 at 4 GPUs
N_SAMPLES=8
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES))   # 256 at 4 GPUs
MAX_NUM_BATCHED_TOKENS=$((8192 * NUM_GPUS))

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   --load ${VIME_CKPT}
   --save ${VIME_CKPT}
   --save-interval 500
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --apply-chat-template
   --metadata-key metadata
   --rollout-shuffle

   # Fully-async rollout WITH the AI21 seams (the stock fully_async_rollout
   # path skips the dynamic filter + all-samples hook entirely).
   --rollout-function-path vime_plugins.rollout.fully_async_ai21.generate_rollout_fully_async_ai21
   # eval defaults to the rollout function path, which raises on eval — pin
   # eval back to the stock synchronous implementation.
   --eval-function-path vime.rollout.vllm_rollout.generate_rollout

   # AI21 evaluators reward
   --custom-rm-path vime_plugins.rm.ai21_evaluators.ai21_reward

   # Curriculum: drop zero-variance groups + snooze easy prompts
   --dynamic-sampling-filter-path vime_plugins.filters.snoozing.snoozing_filter
   --over-sampling-batch-size ${ROLLOUT_BATCH_SIZE}

   # AI21 observability metrics (see the v2 script for the full breakdown)
   --rollout-all-samples-process-path vime_plugins.metrics.rollout_metrics.capture_prefilter_metrics
   --custom-rollout-log-function-path vime_plugins.metrics.rollout_metrics.ai21_rollout_log

   --num-rollout 15
   --num-epoch 1
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 2048
   --rollout-temperature 1.0
   --rollout-top-k 200

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

ASYNC_ARGS=(
   # Push fresh weights to the rollout engines every step (train_async.py).
   # Raising this trades policy freshness for fewer weight-sync stalls.
   --update-weights-interval 1
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data regression ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 2048
   --eval-temperature 0.6
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   --eps-clip-c 10
   --entropy-coef 0.0

   # Mismatch correction matters MORE here than in the sync run: fully-async
   # data is off-policy, so the TIS/RS weights are doing real work.
   --use-tis
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
   --custom-config-path scripts/ai21-regression-mis.yaml
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --clip-grad 1.0
   --accumulate-allreduce-grads-in-fp32
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1
   --vllm-gpu-memory-utilization 0.85
   --vllm-max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}
   --vllm-max-num-seqs 128
   --vllm-enable-prefix-caching
   # In-flight pool size = --vllm-server-concurrency * num_engines (default is
   # usually fine; raise it if rollout GPUs sit idle between steps).
)

WANDB_ARGS=()
if [ -n "${WANDB_KEY:-}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-key ${WANDB_KEY}
      --wandb-project ${WANDB_PROJECT:-regression}
      --wandb-group ${WANDB_GROUP:-qwen2.5-3B-ai21-async}
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---- Ray head + job submit ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${VIME_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"AI21_EVALUATORS_TIMEOUT\": \"${AI21_EVALUATORS_TIMEOUT}\",
    \"AI21_CLEAN_THINKING_TRACE\": \"${AI21_CLEAN_THINKING_TRACE}\",
    \"AI21_SNOOZE_NUM_TIMES\": \"${AI21_SNOOZE_NUM_TIMES}\",
    \"AI21_SNOOZE_MEAN_SCORE_THRESHOLD\": \"${AI21_SNOOZE_MEAN_SCORE_THRESHOLD}\",
    \"AI21_SNOOZE_ID_KEY\": \"${AI21_SNOOZE_ID_KEY}\",
    \"AI21_FILTERED_ROLLOUT_DUMP_PATH\": \"${AI21_FILTERED_ROLLOUT_DUMP_PATH}\",
    \"AI21_CONFIG_DUMP_PATH\": \"${AI21_CONFIG_DUMP_PATH}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${ASYNC_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
