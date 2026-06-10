export WANDB_API_KEY=34c70f7b56bd8cc9235fac9991c7a84296139298

mkdir -p /root/models /root/datasets /root/vime_lora_runs

hf download Qwen/Qwen3-8B \
  --local-dir /root/models/Qwen3-8B

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/datasets/dapo-math-17k

source scripts/models/qwen3-8B.sh

if [ ! -f /root/models/Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt ]; then
  PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 1 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/models/Qwen3-8B \
    --save /root/models/Qwen3-8B_torch_dist
else
  echo "Skip checkpoint conversion: /root/models/Qwen3-8B_torch_dist already exists"
fi

ray stop --force || true
pkill -9 ray || true
pkill -9 -f '[v]llm serve|VLL[M]::' || true

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{
    "env_vars": {
      "PYTHONPATH": "/root/Megatron-LM/",
      "CUDA_DEVICE_MAX_CONNECTIONS": "1",
      "MASTER_ADDR": "127.0.0.1"
    }
  }' \
  -- python3 train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/models/Qwen3-8B \
  --ref-load /root/models/Qwen3-8B_torch_dist \
  --load /root/models/Qwen3-8B \
  --finetune \
  --no-load-optim \
  --no-load-rng \
  --save /root/vime_lora_runs/qwen3-8b-dapo-lora-smoke \
  --save-interval 1 \
  --megatron-to-hf-mode bridge \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --colocate \
  --calculate-per-token-loss \
  --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 1024 \
  --rollout-temperature 0.8 \
  --global-batch-size 1 \
  --micro-batch-size 1 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.00 \
  --kl-loss-type k1 \
  --kl-coef 0.00 \
  --entropy-coef 0.00 \
  --eps-clip 4e-4 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 4096 \
  --rollout-num-gpus-per-engine 1 \
  --rollout-num-gpus 1 \
  --vllm-gpu-memory-utilization 0.45 \
  --vllm-max-cudagraph-capture-size 32 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --lora-type lora \
  --target-modules all-linear \
  --only-train-params-name-list lora_A lora_B linear_in linear_out \
  --lora-adapter-name vime_lora \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --use-wandb \
  --wandb-host https://api.wandb.ai \
  --wandb-project vime-lora-smoke \
  --wandb-group qwen3-8b-dapo-single-h100 \
  --wandb-key "${WANDB_API_KEY}" \
  --disable-wandb-random-suffix