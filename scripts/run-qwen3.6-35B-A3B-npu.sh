#!/bin/bash
# for rerun the task
pkill -9 -f '[v]llm serve|VLL[M]::'
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export DISABLE_L2_CACHE=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_AOT_COMPILE=0
export PYTHONPATH="/root/Megatron-LM/:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_BUFFSIZE=512
export HCCL_BUFFSIZE_EP=512

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

source /root/vime/scripts/models/qwen3.5-35B-A3B.sh

CKPT_ARGS=(
   --hf-checkpoint /path/to/Qwen3.6-35B-A3B
   --load /path/to/Qwen3.6-35B-A3B
   --ref-load /path/to/Qwen3.6-35B-A3B
   --save /path/to/Qwen3.6-35B-A3B_vime_npu_colocate/
   --save-hf /path/to/Qwen3.6-35B-A3B_vime_npu_colocate_hf/
   --no-save-optim
   --save-interval 25
)

ROLLOUT_ARGS=(
   --prompt-data /path/to/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 14336
   --rollout-max-prompt-len 2048
   --rollout-temperature 1
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 25
   --eval-prompt-data aime /path/to/aime-2024/aime-2024.jsonl@[0:250]
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 18
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --moe-token-dispatcher-type alltoall

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   # --qkv-format bshd
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --log-probs-chunk-size 1024
   --data-pad-size-multiplier 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
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
   --vllm-additional-config '{"weight_nz_mode":0}'
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.7
   --vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256)
   --vllm-enable-expert-parallel
   --vllm-data-parallel-size 2
   # --vllm-enforce-eager
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   --use-flash-attn
   --no-gradient-accumulation-fusion
)

ray start --head --num-gpus 0 --resources '{"NPU": 16}' --node-ip-address 127.0.0.1 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

sleep 5

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 16 \
   --colocate \
   --train-memory-margin-bytes 8589934596 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PROFILE_ARGS[@]} \
   2>&1 | tee /path/to/vime_qwen36_35b_a3b_npu_colocate.log
