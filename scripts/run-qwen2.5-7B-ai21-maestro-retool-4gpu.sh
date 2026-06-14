#!/bin/bash
#
# vime port of ai21-verl's exp_regression_maestro_retool.py :: `qwen_25_7b_4gpu`
# (Qwen2.5-7B retool-SFT, GRPO on DAPO-math via a multi-turn Python-code agent loop).
#
#   verl equiv: poetry run train grpo exp_regression_maestro_retool --config qwen_25_7b_4gpu
#
# WHAT'S BEING REPRODUCED
#   The maestro agent loop (make_multistep_action_agent_loop_config): each rollout the
#   policy may call an `execute_python_code` HTTP tool up to max_steps=10 times, reading
#   stdout back between turns, until it emits a final \boxed{} answer. vime has no maestro,
#   but it already supports multi-turn/tool rollouts via the custom-generate seam:
#     vime_plugins/rollout/multi_turn_agent.py  — owns token accounting / loss masking / budgets
#     vime_plugins/rollout/retool_agent.py       — the per-turn code-exec step (parse -> run -> observe)
#
#   The code runner is the SAME contract as maestro: an HTTP service taking {"code": ...}
#   at a URL. Maestro injected CodeRunnerService.get_url() as CODE_RUNNER_URL; that URL can
#   point anywhere — a deployed eval-code-runner OR an instance you run in your debug pod.
#   Set CODE_RUNNER_URL below.
#
# Prereqs:
#   1. retool-SFT HF checkpoint at $HF_CKPT (qwen-25-7b-retool-sft; Qwen2.5-7B arch).
#   2. Megatron torch_dist checkpoint at $REF_LOAD (convert_hf_to_torch_dist.py, as in the
#      other run scripts, using scripts/models/qwen2.5-7B.sh MODEL_ARGS).
#   3. A reachable code runner:  export CODE_RUNNER_URL=http://<host>:<port>/run_code
#   4. Train/val amalgam mixes flattened to JSONL (see DATA PREP below).
#   5. AI21 deps installed: bash scripts/install_ai21_deps.sh
#
# Run:  CODE_RUNNER_URL=http://localhost:8000/run_code bash scripts/run-qwen2.5-7B-ai21-maestro-retool-4gpu.sh

set -ex

# ---- rerun cleanup (mirrors the other run-*.sh scripts) ----
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
pkill -9 -f 'VLLM' || true
pkill -9 -f 'vllm' || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 || true
fi
sleep 2

export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Cluster sizing  (verl Resources(nodes=1, gpus_per_node=4, vllm_tensor_parallel_size=1))
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS}}
[ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ] && NUM_GPUS=4   # maestro 4gpu config
echo "NUM_GPUS: $NUM_GPUS"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh"   # provides MODEL_ARGS (Qwen2.5-7B arch)

# ---------------------------------------------------------------------------
# Paths  (verl model_name=qwen-25-7b-retool-sft)
# ---------------------------------------------------------------------------
HF_CKPT=${HF_CKPT:-/root/qwen-25-7b-retool-sft}
REF_LOAD=${REF_LOAD:-/root/qwen-25-7b-retool-sft_torch_dist}
VIME_CKPT=${VIME_CKPT:-/root/qwen-25-7b-retool-sft_vime}

# Code runner endpoint (maestro CODE_RUNNER_URL). Override to your debug-pod / deployed runner.
export CODE_RUNNER_URL=${CODE_RUNNER_URL:-http://localhost:8000/run_code}

# ---------------------------------------------------------------------------
# DATA PREP — verl data were GCS amalgam mixes:
#   train="gs://ai21-algo-studio-research/uriyap/retool/dapo_math_17k_amalgam_train.json"
#   val  ="gs://ai21-algo-studio-research/uriyap/retool/dapo_aime_val_mix.json"
# vime loads ONE --prompt-data JSONL, so flatten each mix offline (same converter the
# regression-v2 script uses; rows carry metadata.reward_model for the ai21 reward):
#   python -m vime_plugins.data.amalgam_to_jsonl \
#     --mix gs://ai21-algo-studio-research/uriyap/retool/dapo_math_17k_amalgam_train.json \
#     --out /root/dapo_math_17k_train.jsonl --input-key prompt --metadata-key metadata
#   python -m vime_plugins.data.amalgam_to_jsonl \
#     --mix gs://ai21-algo-studio-research/uriyap/retool/dapo_aime_val_mix.json \
#     --out /root/dapo_aime_val.jsonl --input-key prompt --metadata-key metadata
# verl val_examples_limit=160 -> trim the val JSONL to 160 rows (e.g. `head -n 160`).
# ---------------------------------------------------------------------------
TRAIN_DATA=${TRAIN_DATA:-/root/dapo_math_17k_train.jsonl}
EVAL_DATA=${EVAL_DATA:-/root/dapo_aime_val.jsonl}

# ---------------------------------------------------------------------------
# AI21 reward (amalgam/VerifiableTask data -> ai21 evaluators reward reads metadata.reward_model)
# ---------------------------------------------------------------------------
export AI21_EVALUATORS_TIMEOUT=${AI21_EVALUATORS_TIMEOUT:-120.0}
export AI21_CLEAN_THINKING_TRACE=${AI21_CLEAN_THINKING_TRACE:-true}

# ---------------------------------------------------------------------------
# Batch sizing — `qwen_25_7b_4gpu` overrides (nodes=1):
#   batch.train_batch_size = 16*4 = 64   -> --rollout-batch-size 64
#   rollout.n_train        = 8           -> --n-samples-per-prompt 8
#   batch.ppo_mini_batch_size = 2*4 = 8 prompts -> per optimizer step = 8 prompts * 8 samples
#                                                  => --global-batch-size 64 (8 PPO updates/rollout)
#   batch.micro_batch_max_tokens = 12*1024 = 12288 -> dynamic packing --max-tokens-per-gpu 12288
# ---------------------------------------------------------------------------
ROLLOUT_BATCH_SIZE=64
N_SAMPLES=8
GLOBAL_BATCH_SIZE=64

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   --load ${VIME_CKPT}
   --save ${VIME_CKPT}
   --save-interval 50              # verl trainer.save_freq=50
)

ROLLOUT_ARGS=(
   --prompt-data ${TRAIN_DATA}
   --input-key prompt
   --apply-chat-template
   --metadata-key metadata         # carries reward_model for the ai21 reward
   --rollout-shuffle

   # ---- multi-turn code-execution agent loop (the maestro reproduction) ----
   --custom-generate-function-path vime_plugins.rollout.multi_turn_agent.generate
   --custom-config-path scripts/maestro-retool-agent.yaml   # agent_step_path, agent_max_turns=10, retool_*

   # AI21 evaluators reward (verl data are amalgam VerifiableTasks; reward via metadata.reward_model)
   --custom-rm-path vime_plugins.rm.ai21_evaluators.ai21_reward

   --num-rollout 50                # verl trainer.total_training_steps=50
   --num-epoch 1
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES}            # verl rollout.n_train=8

   # Token budgets: maestro max_tokens=2048/step, max_prompt=8192, total_token_budget=12288.
   # The scaffold caps each turn at --rollout-max-response-len and the whole episode
   # (prompt + all turns) at --rollout-max-context-len.
   --rollout-max-prompt-len 8192   # verl data.max_prompt_length = 8*1024
   --rollout-max-response-len 2048 # maestro per-step max_tokens
   --rollout-max-context-len 20480 # prompt(8192) + total_token_budget(12288)
   --rollout-temperature 1.0       # maestro temperature=1.0

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

EVAL_ARGS=(
   # verl val_before_train=True (vime evals before train by default); no test_freq set on the
   # base qwen_25_7b config -> eval at the final step here. Adjust --eval-interval to taste.
   --eval-interval 50
   --eval-prompt-data dapo_aime ${EVAL_DATA}
   --n-samples-per-eval-prompt 4   # verl rollout.n_val=4
   --eval-max-response-len 2048
   --eval-max-context-len 20480
   # verl val_examples_limit=160 -> trim the eval JSONL to 160 rows (see DATA PREP).
)

PERF_ARGS=(
   # 7B at TP=1, data-parallel across 4 GPUs (verl vllm_tensor_parallel_size=1).
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full    # verl enable_gradient_checkpointing
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size        # verl micro_batch_max_tokens (token-packed micro batches)
   --max-tokens-per-gpu 12288      # verl batch.micro_batch_max_tokens = 12*1024
)

GRPO_ARGS=(
   --advantage-estimator grpo      # verl algorithm.adv_estimator=grpo
   --eps-clip 0.2
   --entropy-coef 0.0
   # verl ppo.importance_sampling_correction=False -> NO TIS (--use-tis intentionally omitted).
   # verl did not enable use_kl_loss for this experiment -> KL loss off.
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                       # verl training.learning_rate=1e-6
   --lr-decay-style constant
   --weight-decay 0.01             # verl training.weight_decay=0.01
   --clip-grad 1.0                 # verl training.grad_clip=1.0
   --accumulate-allreduce-grads-in-fp32
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1             # verl vllm_tensor_parallel_size=1
   --vllm-gpu-memory-utilization 0.75          # verl 4gpu rollout.gpu_memory_utilization=0.75
   --vllm-max-num-batched-tokens 16384         # verl 4gpu max_num_batched_tokens = 4096*4
   --vllm-max-num-seqs 128                     # verl 4gpu rollout_max_num_seqs = 32*4
   --vllm-enable-prefix-caching                # verl rollout.enable_prefix_caching=True
)
# NOTE: verl agent_rollout.num_workers=8*4=32 is maestro's per-loop worker concurrency. vime
# bounds in-flight rollout requests via the router / generate semaphore rather than a worker
# count, so there is no 1:1 knob — concurrency is governed by --vllm-server-concurrency + GPUs.

WANDB_ARGS=()
if [ -n "${WANDB_KEY:-}" ]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-key ${WANDB_KEY}
      --wandb-project ${WANDB_PROJECT:-vime}
      --wandb-group ${WANDB_GROUP:-qwen2.5-7B-ai21-maestro-retool-4gpu}
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
#   - maestro action_model/orchestration (quality_threshold, MaestroBudget prompt/completion
#     token caps) are replaced by the simpler multi_turn_agent loop: per-turn cap +
#     episode context cap + max_turns. The prompt/completion 100k MaestroBudget is far above
#     a single episode and is not separately enforced.
#   - num_workers (maestro loop concurrency) has no vime equivalent (see VLLM_ARGS note).
#   - Observation wrapping format is a guess that MUST match the retool-SFT model's training
#     transcript shape; tune retool_observation_format in scripts/maestro-retool-agent.yaml.
# ---------------------------------------------------------------------------

# ---- Ray head + job submit ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${VIME_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"CODE_RUNNER_URL\": \"${CODE_RUNNER_URL}\",
    \"AI21_EVALUATORS_TIMEOUT\": \"${AI21_EVALUATORS_TIMEOUT}\",
    \"AI21_CLEAN_THINKING_TRACE\": \"${AI21_CLEAN_THINKING_TRACE}\"
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
