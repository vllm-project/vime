#!/usr/bin/env bash

set -euo pipefail

cd /root/vime

RUN_ROOT="/host/rl-runs/rl-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/trace"
ln -sfn "${RUN_ROOT}" /work/runs/latest

export PYTHONUNBUFFERED=1
# --- ROCm (gfx950) ---------------------------------------------------------
# Clear baked NVTE_* so Megatron honours --attention-backend flash.
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN
# Let Ray inherit our device mask instead of rewriting it.
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export MASTER_ADDR=127.0.0.1
export SWE_AGENT=claude_code
export SWE_TRAIN_PROTOCOL=scaleswe
export ADAPTER_PUBLIC_HOST=127.0.0.1
export ADAPTER_BIND_HOST=0.0.0.0
export ADAPTER_PORT=18001
export SWE_BOOT_CONCURRENCY=1
export SWE_BOOT_RETRIES=1
export SWE_AGENT_TIME_BUDGET_SEC=600
export SWE_EVAL_TIMEOUT_SEC=300
export SWE_ROLLOUT_GUARD_SEC=9000
export VIME_AGENT_NODE_TARBALL=/work/assets/node-v22.20.0-linux-x64.tar.xz
export VIME_AGENT_CC_TARBALL=/work/assets/anthropic-ai-claude-code.tgz
export VIME_MAX_TURNS_PER_SID="${MAX_TURNS:-80}"
export VIME_AGENT_CC_EXTRA_ENVS='{"ANTHROPIC_MODEL":"claude-sonnet-4-5"}'
export VIME_AGENT_CC_EXTRA_ARGS="--disable-slash-commands --disallowedTools Agent WebFetch WebSearch Write NotebookEdit Workflow ScheduleWakeup SendMessage ListAgents ReportFindings CronCreate CronDelete CronList EnterWorktree ExitWorktree TaskCreate TaskUpdate TaskList TaskGet TaskStop TaskOutput"
export VLLM_DEEP_GEMM_WARMUP=skip
export SWE_CC_PROMPT="Complete the issue in PROBLEM_STATEMENT.md. Inspect the relevant source, actually edit the smallest possible source-only fix, and run a focused behavior check. Do not edit tests or commit, and do not merely describe a patch. Finish with a one-line summary."
export VIME_LOCAL_SANDBOX_TRACE_DIR="${RUN_ROOT}/trace"
export no_proxy=127.0.0.1
export NO_PROXY=127.0.0.1

source scripts/models/${MODEL_CONF:-qwen3-4B}.sh

ray stop --force || true
for pat in "VLLM::" "EngineCore" "ray::" "train.py" \
           "raylet|gcs_server|ray/dashboard|default_worker|log_monitor|runtime_env_agent|autoscaler"; do
  pkill -9 -f "$pat" || true
done
sleep 3
pkill -9 -f "VLLM::" || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NGPU:-8} --disable-usage-stats --dashboard-host=0.0.0.0

RUNTIME_ENV_JSON=$(python - <<'PY'
import json
import os

prefixes = ("ADAPTER_", "SWE_", "VIME_", "VLLM_")
env = {
    key: value
    for key, value in os.environ.items()
    if key.startswith(prefixes) or key in {"CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                                           "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
                                           "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
                                           "MASTER_ADDR", "NO_PROXY", "no_proxy"}
}
env.update(
    PYTHONUNBUFFERED="1",
    PYTHONPATH="/root/vime:/root/Megatron-LM",
    NCCL_NVLS_ENABLE="0",  # AMD: no NVLink SHARP
)
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python -u train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --ref-load "${MODEL_DIR}" \
  --custom-generate-function-path agent_run.local_multi_turn_smoke.generate.generate \
  --prompt-data /work/tasks/${TASKS:-sympy-10.jsonl} \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --num-rollout ${NUM_ROLLOUT:-2} \
  --rollout-batch-size ${RB:-4} \
  # n-samples-per-prompt and rollout-temperature are both load-bearing for GRPO:
  # the advantage is computed within a prompt's sample group, so one sample -- or
  # n identical greedy samples -- gives an advantage of exactly zero and no
  # gradient. Changing only one of the two does not help. Measured: at n=4 this
  # task set showed 2/10 prompts with usable variance, at n=8 it showed 4/10.
  --n-samples-per-prompt ${N_SAMPLES:-8} \
  --rollout-max-context-len ${CTX:-32768} \
  --rollout-max-response-len ${RESP:-4096} \
  --rollout-stop-token-ids 151645 151643 \
  # temperature > 0 and n-samples-per-prompt > 1 are both load-bearing for GRPO:
  # the advantage is computed within a prompt's sample group, so a single sample
  # (or n identical greedy samples) gives an advantage of exactly zero and no
  # gradient. Changing only one of the two does not help.
  --rollout-temperature 1.0 \
  --num-steps-per-rollout 1 \
  # Not a free parameter: vime asserts
  #   global_batch_size == rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
  # Override GB whenever RB or N_SAMPLES changes, or startup fails validation.
  --global-batch-size ${GB:-32} \
  --micro-batch-size 1 \
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt" \
  --load  "${CKPT_DIR:-/work/runs/ckpt}" \
  --save  "${CKPT_DIR:-/work/runs/ckpt}" \
  --save-interval 100000 \
  --tensor-model-parallel-size ${TP:-1} \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu ${MTPG:-32768} \
  --log-probs-chunk-size 1024 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.0 \
  # Without an entropy bonus the policy collapses at any usable learning rate.
  --entropy-coef ${ENT:-0.01} \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  # 1e-6 left the policy effectively static (11 steps, trend t=-0.21); 1e-5
  # collapsed entropy 0.345 -> 0.096 within 2 steps and reward regressed after
  # an initial rise. 3e-6 with the entropy bonus held entropy flat-to-rising
  # across 30 steps while reward rose from 0.254 to 0.596 (t=+4.19).
  --lr ${LR:-3e-6} \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus ${NGPU:-8} \
  --rollout-num-gpus-per-engine ${TP:-1} \
  --vllm-gpu-memory-utilization "${VLLM_MEM_UTIL:-0.60}" \
  --update-weight-transport disk \
  --update-weight-disk-dir /work/runs/wsync \
  --vllm-max-num-seqs "${MAX_SEQS:-8}" \
  --vllm-max-num-batched-tokens "${MAX_BT:-4096}" \
  --vllm-tool-call-parser hermes \
  --vllm-reasoning-parser qwen3 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --no-gradient-accumulation-fusion \
  --no-offload-train \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${NGPU:-8} \
  --colocate \
  2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}" | tee "${RUN_ROOT}/completed.txt"
