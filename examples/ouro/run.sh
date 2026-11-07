#!/usr/bin/env bash
set -euo pipefail
model=${1:?HF Ouro directory}
data=${2:?Training JSONL}
output=${3:?Run directory}
updates=${4:?Number of updates}
shift 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
gpus=${GPUS_PER_NODE:-2}
mkdir -p "$output"
if [ "$gpus" -eq 1 ]; then
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29500}
  launcher=(python)
else
  launcher=(python -m torch.distributed.run --standalone --nproc_per_node="$gpus")
fi
"${launcher[@]}" "$(dirname "$0")/train.py" \
 --debug-train-only --actor-num-nodes 1 --actor-num-gpus-per-node "$gpus" --num-gpus-per-node "$gpus" \
 --custom-model-provider-path vime_plugins.ouro.model.model_provider \
 --hf-checkpoint "$model" --load "$model" --ref-load "$model" \
 --num-layers 24 --hidden-size 2048 --ffn-hidden-size 5632 --num-attention-heads 16 \
 --kv-channels 128 --group-query-attention --num-query-groups 16 \
 --normalization RMSNorm --norm-epsilon 1e-6 --swiglu --disable-bias-linear \
 --untie-embeddings-and-output-weights --position-embedding-type rope --rotary-base 1000000 \
 --vocab-size 49152 --seq-length 4096 --max-position-embeddings 65536 \
 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
 --save-interval "$updates" --save "$output/checkpoint" --ouro-run-dir "$output" \
 --prompt-data "$data" --input-key prompt --label-key label \
 --num-rollout "$updates" --rollout-batch-size 4 --n-samples-per-prompt 4 --global-batch-size 16 \
 --num-steps-per-rollout 1 --micro-batch-size 1 --rollout-max-response-len 512 \
 --rollout-temperature 1 --rollout-top-p 1 --advantage-estimator grpo --use-rollout-logprobs \
 --kl-coef 0 --entropy-coef 0 --eps-clip .2 --eps-clip-high .28 \
 --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay .1 --adam-beta1 .9 --adam-beta2 .98 \
 --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
 --no-gradient-accumulation-fusion --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
 --transformer-impl local --no-rope-fusion \
 --seed 42 "$@"
