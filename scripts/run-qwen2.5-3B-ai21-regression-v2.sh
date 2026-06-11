#!/bin/bash
#
# vime port of ai21-verl's examples/grpo_trainer/ai21_regression_v2.sh
# (Qwen2.5-3B-Instruct GRPO regression, AI21 features).
#
# This re-homes the verl Hydra experiment onto vime's CLI + the migrated
# vime_plugins/ seams. Mapping of every verl override is inline below.
#
# Prereqs:
#   1. HF checkpoint at $HF_CKPT (Qwen2.5-3B-Instruct).
#   2. Megatron torch_dist checkpoint at $REF_LOAD — produced by
#        PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#          ${MODEL_ARGS[@]} --hf-checkpoint $HF_CKPT --save $REF_LOAD
#      (this is the conversion you already ran).
#   3. Train/val data merged to JSONL (see "DATA PREP" below).
#   4. AI21 deps installed: bash scripts/install_ai21_deps.sh
#
# Run:  bash scripts/run-qwen2.5-3B-ai21-regression-v2.sh

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
# Cluster sizing
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
[ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ] && NUM_GPUS=4   # verl regression default = 4 GPUs
echo "NUM_GPUS: $NUM_GPUS"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-3B.sh"   # provides MODEL_ARGS (Qwen2.5-3B arch)

# ---------------------------------------------------------------------------
# Paths  (verl: MODEL_PATH=/dev/shm/qwen2_5_3b downloaded from GCS)
# ---------------------------------------------------------------------------
HF_CKPT=${HF_CKPT:-/root/Qwen2.5-3B-Instruct}
REF_LOAD=${REF_LOAD:-/root/Qwen2.5-3B-Instruct_torch_dist}   # your converted torch_dist ckpt
VIME_CKPT=${VIME_CKPT:-/root/Qwen2.5-3B-Instruct_vime}

# ---------------------------------------------------------------------------
# DATA PREP — verl used AmalgamDataset over a GCS mix-spec JSON.
# vime loads ONE --prompt-data JSONL, so flatten the mix offline first.
# Each row gets metadata.reward_model / aggregation_config / id for the
# ai21-evaluators reward (vime_plugins/data/amalgam_to_jsonl.py).
#
#   python -m vime_plugins.data.amalgam_to_jsonl \
#     --mix  gs://ai21-mammoth-storage/users/asafk/verl-data-configs/regression_short_train.json \
#     --out  /root/regression_short_train.jsonl \
#     --input-key messages --metadata-key metadata
#   python -m vime_plugins.data.amalgam_to_jsonl \
#     --mix  gs://ai21-mammoth-storage/users/asafk/verl-data-configs/regression_short_test.json \
#     --out  /root/regression_short_test.jsonl \
#     --input-key messages --metadata-key metadata
# ---------------------------------------------------------------------------
TRAIN_DATA=${TRAIN_DATA:-/root/regression_short_train.jsonl}
EVAL_DATA=${EVAL_DATA:-/root/regression_short_test.jsonl}

# ---------------------------------------------------------------------------
# AI21 plugin config — verl passed these as Hydra keys; the migrated plugins
# read them from env (see each plugin's docstring).
# ---------------------------------------------------------------------------
# reward.ai21_evaluators_timeout=120.0 / reward.ai21_clean_thinking_trace=True
export AI21_EVALUATORS_TIMEOUT=${AI21_EVALUATORS_TIMEOUT:-120.0}
export AI21_CLEAN_THINKING_TRACE=${AI21_CLEAN_THINKING_TRACE:-true}
# algorithm.filter_groups.num_times_to_snooze_easy_examples=5 / snooze_mean_score_threshold=0.99
export AI21_SNOOZE_NUM_TIMES=${AI21_SNOOZE_NUM_TIMES:-5}
export AI21_SNOOZE_MEAN_SCORE_THRESHOLD=${AI21_SNOOZE_MEAN_SCORE_THRESHOLD:-0.99}
export AI21_SNOOZE_ID_KEY=${AI21_SNOOZE_ID_KEY:-id}
# Filtered-out rollout dumping (verl #113; was on by default via trainer.rollout_data_dir):
# every group dropped by the snoozing filter is appended here as JSONL.
export AI21_FILTERED_ROLLOUT_DUMP_PATH=${AI21_FILTERED_ROLLOUT_DUMP_PATH:-/dev/shm/generations/rollout_filtered_out.jsonl}
# Resolved-config dump (verl main_ppo always dumped the resolved Hydra config):
# "auto" -> <--save dir>/resolved_config.json, written on the plugins' first call.
export AI21_CONFIG_DUMP_PATH=${AI21_CONFIG_DUMP_PATH:-auto}

# Length-reward shaping (vime_plugins/rm/length_reward.py) is intentionally NOT wired:
# the verl regression set reward.length_reward.enable=False (coeff 0.2 configured but off).
# To enable it, add to ROLLOUT_ARGS:
#   --custom-reward-post-process-path vime_plugins.rm.length_reward.length_reward_post_process
# and export AI21_LENGTH_REWARD_COEFF=0.2

# ---------------------------------------------------------------------------
# Batch sizing (verl regression, scales with GPU count; 4 GPUs == original)
#   data.train_batch_size = 8*GPUS  -> --rollout-batch-size
#   actor_rollout_ref.rollout.n = 8 -> --n-samples-per-prompt
#   ppo_mini_batch_size = 8*GPUS    -> one update over the whole rollout
#                                       => --global-batch-size = batch * n
# ---------------------------------------------------------------------------
ROLLOUT_BATCH_SIZE=$((8 * NUM_GPUS))            # 32 at 4 GPUs
N_SAMPLES=8
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES))   # 256 at 4 GPUs
MAX_NUM_BATCHED_TOKENS=$((8192 * NUM_GPUS))     # vllm max_num_batched_tokens, 32768 at 4 GPUs

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}          # only consumed if KL is on; harmless otherwise
   --load ${VIME_CKPT}
   --save ${VIME_CKPT}
   --save-interval 500             # verl trainer.save_freq=500
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt              # amalgam_to_jsonl wrote the conversation under the 'prompt' column (verl data.prompt_key=messages, but the converter's --input-key default is 'prompt')
   --apply-chat-template
   --metadata-key metadata         # carries reward_model / aggregation_config / id
   --rollout-shuffle

   # AI21 evaluators reward  (verl reward.reward_manager.name=AI21RewardManager)
   --custom-rm-path vime_plugins.rm.ai21_evaluators.ai21_reward
   # ai21_reward returns a plain float by default -> no --reward-key needed.
   # Set --reward-key only if you want the extra fields (score/status/do_exclude)
   # exposed for logging or an exclusion filter; then it returns a dict:
   # --reward-key score   --eval-reward-key score

   # Curriculum: drop zero-variance groups + snooze easy prompts
   # (verl algorithm.filter_groups.enable=True + snooze_* knobs above)
   --dynamic-sampling-filter-path vime_plugins.filters.snoozing.snoozing_filter
   --over-sampling-batch-size ${ROLLOUT_BATCH_SIZE}   # ~ verl filter_groups.max_num_gen_batches budget

   # AI21 observability metrics (port of AI21GRPOTrainer's metric layer):
   #   - capture_prefilter_metrics sees ALL generated groups incl. dropped ones
   #     -> rollout/score/{mean,max,min}/prefilter, constant-score buckets, unsnoozed reward
   #   - ai21_rollout_log merges them + kept-batch reward min/max, logs under rollout/step
   #     (returns False -> additive, vime's default rollout logging still runs)
   --rollout-all-samples-process-path vime_plugins.metrics.rollout_metrics.capture_prefilter_metrics
   --custom-rollout-log-function-path vime_plugins.metrics.rollout_metrics.ai21_rollout_log

   --num-rollout 15                # verl trainer.total_training_steps=15
   --num-epoch 1                   # verl trainer.total_epochs=1
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-prompt-len 2048   # verl data.max_prompt_length=2048
   --rollout-max-response-len 2048 # verl data.max_response_length=2048
   --rollout-temperature 1.0       # verl rollout.temperature=1.0
   --rollout-top-k 200             # verl rollout.top_k=200

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

EVAL_ARGS=(
   # verl trainer.test_freq=5, val_before_train=True (vime evals before train by default)
   --eval-interval 5
   --eval-prompt-data regression ${EVAL_DATA}
   --n-samples-per-eval-prompt 1   # verl rollout.val_kwargs.n=1
   --eval-max-response-len 2048
   --eval-temperature 0.6          # verl rollout.val_kwargs.temperature=0.6
   --eval-top-p 1
)

PERF_ARGS=(
   # 3B fits on one GPU -> TP/PP=1, data-parallel across GPUs (matches verl TP=1)
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   # verl actor_rollout_ref.model.enable_gradient_checkpointing=True
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # verl use_dynamic_bsz=True (ppo_micro_batch_size_per_gpu=16 -> dynamic packing)
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo      # verl algorithm.adv_estimator=grpo
   --eps-clip 0.2                  # verl clip_ratio_low=0.2
   --eps-clip-high 0.28            # verl clip_ratio_high=0.28
   --eps-clip-c 10                 # verl clip_ratio_c=10 (dual-clip)
   --entropy-coef 0.0
   # verl actor.use_kl_loss=False -> KL loss OFF, so --use-kl-loss is intentionally omitted.
   # (verl set kl_loss_coef=0.02 / kl_loss_type=low_var_kl but they were inactive.)

   # Train/infer mismatch correction, both verl features via the mismatch helper:
   #   rollout_correction.rollout_is=token/threshold=1.0  -> TIS truncate @ 1.0
   #   actor.filter_disagreement_logprobs threshold=1.0   -> RS mask outside [e^-1, e^1]
   # Config: scripts/ai21-regression-mis.yaml (keys merged into args by --custom-config-path).
   --use-tis
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
   --custom-config-path scripts/ai21-regression-mis.yaml
   # NOTE: verl loss_agg_mode=seq-mean-token-mean is already vime's DEFAULT reducer
   # (sum_of_sample_mean = per-sequence token-mean, then mean over sequences).
   # Do NOT pass --calculate-per-token-loss (that switches to token-mean).
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6                       # verl actor.optim.lr=5e-06
   --lr-decay-style constant
   --weight-decay 0.01             # verl actor.optim.weight_decay=0.01
   --clip-grad 1.0                 # verl actor.grad_clip=1.0
   --accumulate-allreduce-grads-in-fp32   # ~ verl fp32 master grads (vime keeps bf16 params)
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1             # verl rollout.tensor_model_parallel_size=1
   --vllm-gpu-memory-utilization 0.85          # verl rollout.gpu_memory_utilization=0.85
   --vllm-max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}
   --vllm-max-num-seqs 128                     # verl rollout.max_num_seqs=128
   --vllm-enable-prefix-caching                # verl rollout.enable_prefix_caching=True
)

# verl logged to wandb project "regression". W&B turns on automatically when
# WANDB_KEY is set in the env; do NOT hardcode the key here (the verl script
# leaked one — rotate it). Override via WANDB_PROJECT / WANDB_GROUP (vime derives
# the run name from the group). No network? Add: WANDB_ARGS+=(--wandb-mode offline)
WANDB_ARGS=()
if [ -n "${WANDB_KEY:-}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-key ${WANDB_KEY}
      --wandb-project ${WANDB_PROJECT:-regression}
      --wandb-group ${WANDB_GROUP:-qwen2.5-3B-ai21-regression-v2}
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---------------------------------------------------------------------------
# Known semantic deltas vs verl (accepted):
#   - Disagreement masking: verl recomputed the actor logprob per PPO minibatch and
#     re-derived the mask; vime's TIS hook receives the pre-update (old) actor
#     logprobs, so the mask is fixed per step. Identical on the first minibatch.
#   - actor.fsdp_config.model_dtype=fp32 / cast_params_to_inference_dtype_before_update:
#     vime trains bf16 params + fp32 grad accumulation (--accumulate-allreduce-grads-in-fp32).
#   - the --deterministic mode block (not reproduced).
# ---------------------------------------------------------------------------

# ---- Ray head + job submit (mirrors scripts/run-qwen3-4B.sh) ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Optional: GCS checkpoint sidecar (verl synced checkpoints to GCS).
# Uncomment to mirror --save to a bucket while training runs:
# GCS_DEST=${GCS_DEST:-gs://ai21-mammoth-storage/users/asafk/vime-checkpoints/qwen2.5-3b-regression-v2}
# python -m vime_plugins.checkpoint.gcs_sync watch \
#     --local-dir ${VIME_CKPT} --gcs-dest ${GCS_DEST} --poll-interval 60 --log-stdout &
# GCS_SYNC_PID=$!

# Optional: unified Prometheus /metrics sidecar (vLLM multiproc + Ray; in verl this
# was launched by infra, not the regression script). PROMETHEUS_MULTIPROC_DIR must be
# exported before `ray start` below for vLLM metric files to land there:
# export PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc && mkdir -p ${PROMETHEUS_MULTIPROC_DIR}
# python -m vime_plugins.metrics.prometheus_aggregator --port 9090 --ray-metrics-port 8080 &

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
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
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
   ${MISC_ARGS[@]}
