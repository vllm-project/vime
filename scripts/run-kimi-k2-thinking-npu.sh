#!/bin/bash
set -ex
ulimit -u 65535

# cleanup
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 2
npu-smi info 2>/dev/null | grep rayWorker | awk '{print $4}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# Ray isolation: independent temp-dir, ports, and cleanup
export RAY_TMPDIR=/tmp/ray_vime_npu_kimi_k2_thinking
export RAY_PORT=6379
export RAY_DASHBOARD_PORT=8265
export RAY_AGENT_PORT=52378
unset RAY_ADDRESS RAY_REDIS_ADDRESS

ray stop --force 2>/dev/null || true
rm -rf "${RAY_TMPDIR}"
sleep 2

project_name="vime"
exp_name="kimi_k2_thinking"
RAY_DATA_HOME=${RAY_DATA_HOME:-"/root/logs"}
start_time=$(date +"%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-"${RAY_DATA_HOME}/${project_name}/${exp_name}"}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${start_time}.log"

echo "Experiment Log will be saved to: ${LOG_FILE}"
VIME_DIR="/root/vime"

# NPU environment
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="${VIME_DIR}:/root/Megatron-LM:/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:/root/Megatron-Bridge/src:/root/mbridge:/root/MegatronAdaptor:/root/TransformerEngineNPU:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_DETERMINISTIC=true
export VLLM_ASCEND_ENABLE_NZ=0
export ASCEND_COREDUMP_SIGNAL=None
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0
export TASK_QUEUE_ENABLE=0
export RAY_DISABLE_SIGINT_OVERRIDE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:${LD_LIBRARY_PATH}
export VLLM_DISABLE_COMPILE_CACHE=1
export TRANSFORMERS_VERBOSITY=error
export RUST_LOG=vllm_router_rs=warn

NUM_NPUS=16
source "${VIME_DIR}/scripts/models/kimi-k2-thinking.sh"

CKPT_ARGS=(
   --hf-checkpoint /path/to/Kimi-K2-Thinking
   --ref-load /path/to/Kimi-K2-Thinking
   --load /path/to/Kimi-K2-Thinking_npu/
   --save /path/to/Kimi-K2-Thinking_npu/
   --save-interval 20
   --no-load-optim
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /path/to/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 200
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
    --eval-interval 50
    --eval-prompt-data aime /path/to/aime-2024/aime-2024.jsonl
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 16384
    --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
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
   --rollout-backend vllm
   --rollout-num-gpus-per-engine 16
   --vllm-gpu-memory-utilization 0.7
   --vllm-data-parallel-size 8
   --vllm-enable-experet-parallel
   --vllm-enable-sleep-mode
   --vllm-weight-sync-mode native
   --vllm-enforce-eager
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
)

#MASTER_ADDR
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CURRENT_NODE_IP="${CURRENT_NODE_IP:-$(hostname -I | awk '{print $1}')}"
if [[ "${CURRENT_NODE_IP}" != "${MASTER_ADDR}" ]];then
  # launch the other nodes of ray in container
  unset https_proxy http_proxy proxy
  ray start \
      --address="${MASTER_ADDR}:${RAY_PORT}"
      --node-ip-address "${CURRENT_NODE_IP}" \
      --num-gpus 0 \
      --resources "{\"NPU\": $NUM_NPUS}" \
      --disable-usage-stats \
      --block
else
  # launch the master node of ray in container
  unset https_proxy http_proxy proxy
  ray start --head \
      --temp-dir="${RAY_TMPDIR}" \
      --port="${RAY_PORT}" \
      --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --dashboard-agent-listen-port="${RAY_AGENT_PORT}" \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus 0 \
      --resources "{\"NPU\": $NUM_NPUS}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0

  # Build the runtime environment JSON with proper variable substitution
  RUNTIME_ENV_JSON=$(cat << 'EOF'
  {
    "env_vars": {
      "PYTHONPATH": "${VIME_DIR}:/root/Megatron-LM:/vllm-workspace/vllm:/vllm-workspace/vllm-ascend:/root/Megatron-Bridge/src:/root/mbridge:/root/MegatronAdaptor:/root/TransformerEngineNPU:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
      "CUDA_DEVICE_MAX_CONNECTIONS": "1",
      "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
      "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
      "HCCL_CONNECT_TIMEOUT": "7200",
      "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
      "VLLM_DISABLE_COMPILE_CACHE": "1",
      "TRANSFORMERS_VERBOSITY": "error",
      "RUST_LOG": "vllm_router_rs=warn",
      "ASCEND_CUSTOM_OPP_PATH": "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer",
      "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib"
    }
  }
  EOF
  )

  ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     --working-dir="${VIME_DIR}" \
     -- python3 -u train.py \
     --train-backend megatron \
     --actor-num-nodes 16 \
     --actor-num-gpus-per-node 16 \
     --colocate \
     --update-weight-buffer-size $(( 4 * 512 * 1024 * 1024 ))
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${ROLLOUT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${GRPO_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${EVAL_ARGS[@]} \
     ${VLLM_ARGS[@]} \
     ${MISC_ARGS[@]} \
     2>&1 | tee "${LOG_FILE}"
fi