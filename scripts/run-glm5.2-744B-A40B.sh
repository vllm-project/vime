#!/bin/bash

# GLM-5.2 744B-A40B RL training on 32 nodes / 256 H100 GPUs with PD disaggregation.

# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm5.2-744B-A40B.sh"

if [ -z "${BASE_DIR:-}" ]; then
  echo "BASE_DIR is not set. Please set it to a shared path visible from every node."
  exit 1
fi

SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}

CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/GLM-5.2-FP8
   --ref-load $BASE_DIR/GLM-5.2_torch_dist
   --load $BASE_DIR/GLM-5.2_vime
   --save $BASE_DIR/GLM-5.2_vime
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 65536
   --rollout-temperature 1.0

   --global-batch-size 64
)

# TP=4, PP=8, CP=8 consumes all 256 GPUs (32 nodes) for one training group; DP=1.
# Experts use EP=32: expert_tp(1) * ep(32) * pp(8) = 256 = world_size (expert_dp=1).
#
# DSA cross-layer index sharing requires every pipeline stage to START on a
# "computing" layer (index_topk_freq=4, index_skip_topk_offset=3 -> computing
# layers are 1,2,3,7,11,...,75). A uniform 78/8 split would start stages on skip
# layers and fail. We instead use first=14, last=16, leaving 6 middle stages of
# (78-14-16)/6 = 8 layers each. Stage starts land on global layers
# 1,15,23,31,39,47,55,63 -- all computing layers.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 8
   --decoder-first-pipeline-num-layers 14
   --decoder-last-pipeline-num-layers 16
   --context-parallel-size 8
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --data-pad-size-multiplier 1024
   --log-probs-chunk-size 16384
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
   --tis-clip-low 0.5
   --tis-clip 2.0
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project vime-dev
   # --wandb-group glm5.2-744B-A40B
)

VLLM_CONFIG_FILE=$(mktemp /tmp/vllm_glm52_744B_A40B_XXXXXX.yaml)
# PD disaggregation: 1 prefill engine (64 GPU) + 3 decode engines (192 GPU) = 256.
# Each engine spans 64 GPUs (EP=64, within DeepEP's supported rank set). Prefill
# uses the auto DeepEP path; decode uses low_latency + deep_gemm for throughput.
cat > "${VLLM_CONFIG_FILE}" <<CFG
vllm:
  - name: default
    server_groups:
      - worker_type: prefill
        num_gpus: 64
        num_gpus_per_engine: 64
        overrides:
          # vLLM EngineArgs (sglang ServerArgs translated per §5.5). dp_size->data_parallel_size,
          # ep_size->enable_expert_parallel, chunked_prefill_size->max_num_batched_tokens,
          # max_running_requests->max_num_seqs, deepep_mode:auto->all2all_backend:deepep_high_throughput.
          # Dropped sglang-only: enable_dp_attention / enable_dp_lm_head / moe_dense_tp_size /
          # load_balance_method (no vLLM equivalent).
          data_parallel_size: 64
          enable_expert_parallel: true
          max_num_batched_tokens: 131072
          max_num_seqs: 512
          all2all_backend: deepep_high_throughput
      - worker_type: decode
        num_gpus: 192
        num_gpus_per_engine: 64
        overrides:
          # deepep_mode:low_latency->all2all_backend:deepep_low_latency (§5.5: vLLM has no
          # 'auto'; PD encodes it per-group -- prefill high_throughput, decode low_latency).
          # Dropped sglang-only: enable_dp_attention / enable_dp_lm_head / moe_dense_tp_size /
          # load_balance_method / moe_runner_backend / disable_overlap_schedule / cuda_graph_max_bs.
          data_parallel_size: 64
          enable_expert_parallel: true
          max_num_seqs: 768
          all2all_backend: deepep_low_latency
CFG

# sglang --watchdog-timeout 3600 -> vLLM env (§5.5); no CLI flag for it.
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 64
   --vllm-gpu-memory-utilization 0.70
   --vllm-kv-cache-dtype fp8_e4m3
   --vllm-max-cudagraph-capture-size 8          # was --sglang-cuda-graph-max-bs 8
   --vllm-config "${VLLM_CONFIG_FILE}"

   # MTP / EAGLE speculative decoding using the model's own next-token-prediction
   # layer (GLM-5.2 ships an MTP layer; no separate draft model). sglang's 5
   # --speculative-* flags merge into one vLLM JSON (§5.2): num-draft-tokens 5 ->
   # num_speculative_tokens; num-steps / eagle-topk / draft-attention-backend have
   # no vLLM SpeculativeConfig field.
   --vllm-speculative-config '{"method":"eagle","num_speculative_tokens":5}'

   # NOTE — sglang-coupled args translated/relocated (per knowledge/rl/sglang-to-vllm-
   # translation.md §5.5); this 744B PD script is NOT CI-runnable, so the engine config
   # below is SOP-mapped but hardware-unvalidated:
   #  - dp_size/ep_size/dp-attention/dp-lm-head/moe-dense-tp/max-running-requests and the
   #    DeepEP mode now live in the per-group `overrides:` of $VLLM_CONFIG_FILE above
   #    (deepep_mode auto/low_latency -> all2all_backend deepep_high_throughput/low_latency).
   #  - NSA sparse attn (--sglang-nsa-*-backend / page-size / attention-backend nsa) dropped:
   #    vLLM selects DeepSeek-style sparse attention (sparse_attn_indexer) per the model.
   #  - PD transport (--sglang-disaggregation-transfer-backend mooncake / -ib-device mlx5_1xx)
   #    -> vLLM `--vllm-kv-transfer-config '{"kv_connector":...,"kv_connector_extra_config":
   #    {...}}'`; connector name + IB device list are fabric-specific, configure on target.
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --moe-token-dispatcher-type alltoall
)

if [ -z "${MASTER_ADDR:-}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},10.0.0.0/8,100.64.0.0/10"
export no_proxy="${NO_PROXY_LIST}"
export NO_PROXY="${NO_PROXY_LIST}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [ -n "${HOSTFILE:-}" ]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    if [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]]; then
      continue
    fi
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 -f '[v]llm serve|VLL[M]::' ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "PYTHONPATH": "/root/vime:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUNBUFFERED": "1",
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NET_GDR_LEVEL": "2",
    "NCCL_IB_QPS_PER_CONNECTION": "2",
    "NCCL_IB_TC": "160",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_PXN_DISABLE": "0",
    "NCCL_MIN_CTAS": "4",
    "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "MC_IB_PCI_RELAXED_ORDERING": "1",
    "MLP_SKIP_SORT_RDMA": "true",
    "VLLM_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
    "VLLM_JIT_DEEPGEMM_PRECOMPILE": "true",
    "NVSHMEM_DISABLE_NCCL": "1"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 32 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --no-check-for-nan-in-loss-and-grad \
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 2 )) \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
