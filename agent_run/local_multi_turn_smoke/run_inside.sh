#!/usr/bin/env bash

set -euo pipefail

cd /root/vime

RUN_ROOT="/work/runs/single-agent-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/trace"
ln -sfn "${RUN_ROOT}" /work/runs/latest

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
export VIME_AGENT_CC_EXTRA_ARGS="--disable-slash-commands --disallowedTools Agent WebFetch WebSearch Write NotebookEdit"
export VLLM_DEEP_GEMM_WARMUP=skip
export SWE_CC_PROMPT="Complete the issue in PROBLEM_STATEMENT.md. Inspect the relevant source, actually edit the smallest possible source-only fix, and run a focused behavior check. Do not edit tests or commit, and do not merely describe a patch. Finish with a one-line summary."
export VIME_LOCAL_SANDBOX_TRACE_DIR="${RUN_ROOT}/trace"
export no_proxy=127.0.0.1
export NO_PROXY=127.0.0.1

source scripts/models/qwen3.5-27B.sh

ray stop --force || true
pkill -9 -f '[v]llm serve|VLL[M]::' || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0

RUNTIME_ENV_JSON=$(python - <<'PY'
import json
import os

prefixes = ("ADAPTER_", "SWE_", "VIME_", "VLLM_")
env = {
    key: value
    for key, value in os.environ.items()
    if key.startswith(prefixes) or key in {"CUDA_VISIBLE_DEVICES", "MASTER_ADDR", "NO_PROXY", "no_proxy"}
}
env.update(
    PYTHONUNBUFFERED="1",
    PYTHONPATH="/root/vime:/root/Megatron-LM",
    CUDA_DEVICE_MAX_CONNECTIONS="1",
    NCCL_NVLS_ENABLE="0",
)
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python -u train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /work/models/Qwen3.5-27B-GPTQ-Int4 \
  --ref-load /work/models/Qwen3.5-27B-GPTQ-Int4 \
  --custom-generate-function-path agent_run.local_multi_turn_smoke.generate.generate \
  --prompt-data /work/tasks/sympy-10.jsonl \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --num-rollout 1 \
  --rollout-batch-size 10 \
  --n-samples-per-prompt 1 \
  --rollout-max-context-len 262144 \
  --rollout-max-response-len 16384 \
  --rollout-stop-token-ids 248046 248044 \
  --rollout-temperature 0.0 \
  --num-steps-per-rollout 1 \
  --global-batch-size 10 \
  --micro-batch-size 1 \
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt" \
  --debug-rollout-only \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 262144 \
  --log-probs-chunk-size 1024 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --kl-coef 0.0 \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --rollout-num-gpus 8 \
  --rollout-num-gpus-per-engine 8 \
  --vllm-gpu-memory-utilization 0.80 \
  --vllm-tool-call-parser qwen3_coder \
  --vllm-reasoning-parser qwen3 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}" | tee "${RUN_ROOT}/completed.txt"
