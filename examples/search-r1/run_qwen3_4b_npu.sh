#!/bin/bash
set -ex
ulimit -u 65535

# cleanup
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 2
npu-smi info 2>/dev/null | grep rayWorker | awk '{print $4}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# Ray isolation: independent temp-dir, ports, and cleanup
export RAY_TMPDIR=/tmp/ray_vime_npu_search_r1
export RAY_PORT=6379
export RAY_DASHBOARD_PORT=8265
export RAY_AGENT_PORT=52378
unset RAY_ADDRESS RAY_REDIS_ADDRESS

ray stop --force 2>/dev/null || true
rm -rf "${RAY_TMPDIR}"
sleep 2

project_name="vime"
exp_name="qwen3-4b-search-r1"
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
export PYTHONPATH="${VIME_DIR}/examples/search-r1:/root/Megatron-LM:/root/vllm:/root/vllm-ascend:${VIME_DIR}:/root/Megatron-Bridge:/root/mbridge:/root/MegatronAdaptor:/root/TransformerEngineNPU:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge:${PYTHONPATH}"
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
export TASK_QUEUE_ENABLE=1
export RAY_DISABLE_SIGINT_OVERRIDE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:${LD_LIBRARY_PATH}
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export TRANSFORMERS_VERBOSITY=error
export RUST_LOG=vllm_router_rs=warn

NUM_NPUS=16
source "${VIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint /path/to/Qwen3-4B-Instruct-2507
   --ref-load  /path/to/Qwen3-4B-Instruct-2507
   --load /path/to/Qwen3-4B-Instruct-2507_vime_npu/
   --save /path/to/Qwen3-4B-Instruct-2507_vime_npu/
   --save-interval 100
   --no-load-optim
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /path/to/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data nq_test /path/to/nq_hotpotqa_train/test.parquet@[0:500]
   --n-samples-per-eval-prompt 1
   --eval-input-key prompt
   --eval-label-key reward_model
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # TIS (Trajectory Importance Sampling)
   # --use-tis
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
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.7
   --vllm-enable-sleep-mode
   --vllm-weight-sync-mode native
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search.generate
   --custom-rm-path generate_with_search.reward_func

   # TIS (Trajectory Importance Sampling)
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
unset https_proxy http_proxy proxy
ray start --head \
    --temp-dir="${RAY_TMPDIR}" \
    --port="${RAY_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --dashboard-agent-listen-port="${RAY_AGENT_PORT}" \
    --node-ip-address 127.0.0.1 \
    --num-gpus 0 \
    --resources "{\"NPU\": $NUM_NPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON=$(cat << 'EOF'
{
  "env_vars": {
    "PYTHONPATH": "${VIME_DIR}/examples/search-r1:/root/Megatron-LM:/root/vllm:/root/vllm-ascend:${VIME_DIR}:/root/Megatron-Bridge:/root/mbridge:/root/MegatronAdaptor:/root/TransformerEngineNPU:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
    "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
    "HCCL_CONNECT_TIMEOUT": "7200",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "RUST_LOG": "vllm_router_rs=warn",
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib"
  }
}
EOF
)

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --working-dir="${VIME_DIR}" \
   -- python3 -u train.py \
   --train-backend megatron \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"
