#!/usr/bin/env bash
# Single-node, non-colocated GRPO: 8 actor NPUs and 8 rollout NPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname -- "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/models/qwen3.8-27B.sh"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:-/root}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen3.8-27B}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/datasets/dapo-math-17k}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DATA_ROOT}/checkpoints/Qwen3.8-27B}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT_BASE="${RAY_PORT_BASE:-18479}"
if ! [[ "${RAY_PORT_BASE}" =~ ^[1-9][0-9]*$ ]] || ((RAY_PORT_BASE < 1024 || RAY_PORT_BASE > 64000)); then
    echo "RAY_PORT_BASE must be an integer between 1024 and 64000" >&2
    exit 1
fi

test -f "${MODEL_DIR}/config.json"
test -f "${DATASET_DIR}/dapo-math-17k.jsonl"
mkdir -p "${CHECKPOINT_DIR}"

export PYTHONUNBUFFERED=1 VIME_PLATFORM=npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=900
export VLLM_USE_AOT_COMPILE=0 VLLM_ASCEND_ENABLE_NZ=0 DISABLE_L2_CACHE=1
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
export PYTHONPATH="${REPO_DIR}:/root/Megatron-LM:/root/MegatronAdaptor:/root/TransformerEngineNPU:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Use a separate Ray session and ports; never stop other Ray or Python jobs.
RAY_TEMP_DIR="$(mktemp -d /tmp/vime-qwen38-ray.XXXXXX)"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --port "${RAY_PORT_BASE}" --ray-client-server-port "$((RAY_PORT_BASE + 1))" \
    --dashboard-port "$((RAY_PORT_BASE + 2))" \
    --dashboard-agent-listen-port "$((RAY_PORT_BASE + 3))" \
    --dashboard-agent-grpc-port "$((RAY_PORT_BASE + 4))" \
    --runtime-env-agent-port "$((RAY_PORT_BASE + 5))" \
    --node-manager-port "$((RAY_PORT_BASE + 6))" --object-manager-port "$((RAY_PORT_BASE + 7))" \
    --min-worker-port "$((RAY_PORT_BASE + 100))" --max-worker-port "$((RAY_PORT_BASE + 1099))" \
    --temp-dir "${RAY_TEMP_DIR}" --num-gpus 0 --resources '{"NPU":16}' \
    --disable-usage-stats --dashboard-host 127.0.0.1

ray job submit --address="http://${MASTER_ADDR}:$((RAY_PORT_BASE + 2))" -- \
    python3 train.py \
    "${MODEL_ARGS[@]}" \
    --train-backend megatron --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
    --num-gpus-per-node 16 --rollout-num-gpus 8 --rollout-num-gpus-per-engine 4 \
    --no-offload-train --no-offload-rollout \
    --hf-checkpoint "${MODEL_DIR}" --load "${MODEL_DIR}" --ref-load "${MODEL_DIR}" \
    --save "${CHECKPOINT_DIR}" --save-interval 25 --ckpt-format torch_dist \
    --prompt-data "${DATASET_DIR}/dapo-math-17k.jsonl" --input-key prompt --label-key label \
    --apply-chat-template --rollout-shuffle --rm-type dapo --reward-key score \
    --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size 32 --n-samples-per-prompt 8 \
    --rollout-max-prompt-len 1024 --rollout-max-response-len 4096 --rollout-max-context-len 5120 \
    --rollout-temperature 1 --global-batch-size 256 --balance-data \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 \
    --sequence-parallel --qwen-gdn-tensor-parallel --context-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 4096 --seq-length 5120 \
    --advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 --optimizer-cpu-offload \
    --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
    --vllm-additional-config '{"weight_nz_mode":0}' \
    --vllm-compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --vllm-cudagraph-capture-sizes 8 16 32 64 96 128 \
    --vllm-gpu-memory-utilization 0.85 \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend flash --use-flash-attn \
    --no-gradient-accumulation-fusion
