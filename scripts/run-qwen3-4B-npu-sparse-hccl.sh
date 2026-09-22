#!/bin/bash

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
pkill -9 -f VLLM
sleep 3
ray stop --force
pkill -9 ray; :
pkill -9 python; :
sleep 3
pkill -9 ray; :
pkill -9 python; :
pkill -9 redis; :

set -ex

export PYTHONUNBUFFERED=1
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:?Set ASCEND_RT_VISIBLE_DEVICES explicitly}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
: "${VIME_WORKSPACE_ROOT:?Set VIME_WORKSPACE_ROOT}"
: "${MEGATRON_ADAPTOR_ROOT:?Set MEGATRON_ADAPTOR_ROOT}"
: "${TRANSFORMER_ENGINE_NPU_ROOT:?Set TRANSFORMER_ENGINE_NPU_ROOT}"
export PYTHONPATH="${MEGATRON_ADAPTOR_ROOT}:${TRANSFORMER_ENGINE_NPU_ROOT}:${VIME_WORKSPACE_ROOT}/Megatron-Bridge/src:${VIME_WORKSPACE_ROOT}/Megatron-LM:${PYTHONPATH:-}"

if [[ ! -d "${MEGATRON_ADAPTOR_ROOT}/megatron_adaptor" ]]; then
   echo "MegatronAdaptor package not found under ${MEGATRON_ADAPTOR_ROOT}" >&2
   exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

: "${MODEL_PATH:?Set MODEL_PATH}"
: "${PROMPT_DATA_PATH:?Set PROMPT_DATA_PATH}"
UPDATE_WEIGHT_MODE="${UPDATE_WEIGHT_MODE:-delta}"
UPDATE_WEIGHT_TRANSPORT="${UPDATE_WEIGHT_TRANSPORT:-disk}"
UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-}"
UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR="${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:-/tmp/vime-rollout-checkpoint}"
UPDATE_WEIGHT_MODE="${UPDATE_WEIGHT_MODE:-delta}"
UPDATE_WEIGHT_TRANSPORT="${UPDATE_WEIGHT_TRANSPORT:-disk}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
TRAIN_LR="${TRAIN_LR:-1e-6}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6399}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52365}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-vime-delta}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --load "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --megatron-to-hf-mode bridge
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef "${ENTROPY_COEF}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${TRAIN_LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 2
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
   --vllm-additional-config '{"weight_nz_mode":0}'
)
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
   VLLM_ARGS+=(--vllm-enforce-eager)
fi
if [[ -n "${VIME_UPDATE_WEIGHT_PROFILE_DIR:-}" ]]; then
   # Pass this through VIME's vLLM argument bridge explicitly: Ray runtime-env
   # applies to the trainer, while rollout actors need the profiler config in
   # their own launch command.
   VLLM_ARGS+=(
      --vllm-profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VIME_UPDATE_WEIGHT_PROFILE_DIR}/inference\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"ignore_frontend\":true}"
   )
fi

UPDATE_WEIGHT_ARGS=(
   --update-weight-mode "${UPDATE_WEIGHT_MODE}"
   --update-weight-transport "${UPDATE_WEIGHT_TRANSPORT}"
)
if [[ "${UPDATE_WEIGHT_TRANSPORT}" == "disk" ]]; then
   : "${UPDATE_WEIGHT_DISK_DIR:?Set UPDATE_WEIGHT_DISK_DIR for disk transport}"
   UPDATE_WEIGHT_ARGS+=(
      --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
   )
fi
if [[ "${UPDATE_WEIGHT_MODE}" == "delta" ]]; then
   UPDATE_WEIGHT_ARGS+=(
      --update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
      --update-weight-delta-encoding "${UPDATE_WEIGHT_DELTA_ENCODING:-xor}"
      --update-weight-delta-checksum xxh3-128
   )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --micro-batch-size 1
   --use-flash-attn
)

ray start --head --port="${RAY_GCS_PORT}" --temp-dir="${RAY_TEMP_DIR}" \
--node-ip-address 127.0.0.1 --disable-usage-stats \
--dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
--dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}"

RAY_JOB_ARGS=()
if [[ -n "${VIME_UPDATE_WEIGHT_PROFILE_DIR:-}" ]]; then
   PROFILE_RUNTIME_ENV=$(printf \
      '{"env_vars":{"VIME_UPDATE_WEIGHT_PROFILE_DIR":"%s","VIME_UPDATE_WEIGHT_PROFILE_CALL":"%s","VLLM_VERSION":"%s","VLLM_RPC_TIMEOUT":"%s"}}' \
      "${VIME_UPDATE_WEIGHT_PROFILE_DIR}" \
      "${VIME_UPDATE_WEIGHT_PROFILE_CALL:-3}" \
      "${VLLM_VERSION}" \
      "${VLLM_RPC_TIMEOUT:-1800000}")
   RAY_JOB_ARGS+=(--runtime-env-json "${PROFILE_RUNTIME_ENV}")
fi

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${RAY_JOB_ARGS[@]}" \
-- python3 train.py \
--actor-num-nodes 1 \
--actor-num-gpus-per-node 2 \
--rollout-num-gpus 2 \
${MODEL_ARGS[@]} \
${CKPT_ARGS[@]} \
${ROLLOUT_ARGS[@]} \
${OPTIMIZER_ARGS[@]} \
${GRPO_ARGS[@]} \
${PERF_ARGS[@]} \
${VLLM_ARGS[@]} \
${UPDATE_WEIGHT_ARGS[@]} \
${MISC_ARGS[@]}
