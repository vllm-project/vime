# DSpark Speculative Decoding Training

This example shows how to train a **DSpark** (semi-autoregressive speculative
decoding) draft model alongside the policy model during RL. DSpark drafts
multiple tokens in parallel from intermediate policy hidden states, then vLLM
verifies them in a single forward pass — accelerating rollouts without a
separate draft training pipeline.

## Key Features

- **Online draft training**: The draft model is trained jointly with the policy
  during RL, so it stays in distribution as the policy updates.
- **Weight sync to vLLM**: Draft weights are synced to vLLM every rollout step
  via direct IPC (colocate) or packed tensor transfer (non-colocate), enabling
  immediate speculative decoding in the next rollout.
- **Freeze-policy mode**: Optionally freeze the policy and train only the draft
  model, useful when the RL signal is weak or policy degradation is a concern.

## Prerequisites

1. **Pre-trained DSpark draft checkpoint** (recommended). Training from random
   init converges slowly. Pre-train the draft backbone on a supervised corpus
   first, then pass the checkpoint via `--dspark-pretrained-model`.

2. **Policy model** in both HF and torch_dist formats:

```bash
cd /root/vime
source scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

3. **Dataset** (e.g., dapo-math-17k):

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dspark-pretrained-model` | Path to pre-trained DSpark safetensors. Strongly recommended. |
| `--dspark-block-size` | Number of draft tokens per block (default: 7). |
| `--dspark-num-draft-layers` | Number of decoder layers in the draft backbone (default: 5). |
| `--dspark-target-layer-ids` | Comma-separated policy layer indices to capture hidden states from (default: "1,9,17,25,33"). |
| `--dspark-freeze-policy` | Freeze policy and train only the draft model. |
| `--dspark-ce-loss-alpha` | Weight for cross-entropy loss (default: 0.1). |
| `--dspark-l1-loss-alpha` | Weight for L1/TV loss (default: 0.9). |
| `--dspark-draft-loss-weight` | Weight multiplying draft loss added to policy loss (default: 1.0). |
| `--vllm-speculative-config` | JSON config passed to vLLM. Setting `method: "dspark"` also enables DSpark draft training. |

## Mode Comparison

| Mode | Flag | GPU Layout | Weight Sync |
|------|------|------------|-------------|
| **Colocate** | `--colocate` | Train + rollout share the same GPUs | Direct IPC (fastest) |
| **Non-colocate** | (default) | Train and rollout on separate GPU sets | Packed tensor transfer |

## Running the Example

### Colocate Mode (Recommended)

Train and rollout share the same 8 GPUs. The policy model is offloaded to CPU
during rollout, then restored for the next training step.

```bash
bash examples/dspark/run-qwen3-4B-dspark-colocate.sh
```

GPU layout:

| GPUs | Role |
|------|------|
| 0–7 | Policy Megatron train + vLLM rollout (colocate) |

### Non-Colocate Mode

Train and rollout run on separate GPU groups. This avoids the offload overhead
but requires more GPUs.

```bash
bash examples/dspark/run-qwen3-4B-dspark-non-colocate.sh
```

GPU layout:

| GPUs | Role |
|------|------|
| 0–3 | Policy Megatron train |
| 4–7 | vLLM rollout (with DSpark speculative decoding) |

## What to Expect

On Qwen3-4B with 8x A800 GPUs and a pre-trained DSpark draft checkpoint:

| Metric | Typical Value |
|--------|---------------|
| Draft acceptance rate | 30–40% |
| Mean acceptance length | 3.2–3.8 |
| Rollout speedup | ~2x vs no speculative decoding |
| Weight sync time | ~10s per step |

## FAQ

1. **Do I need a pre-trained draft model?**
   Strongly recommended. Training from random init requires many more steps to
   converge. Pre-train the draft backbone on a supervised corpus, then pass the
   checkpoint via `--dspark-pretrained-model`.

2. **What does `--dspark-freeze-policy` do?**
   It freezes the policy model and trains only the draft model. The policy
   logits are detached so gradients only flow to the draft. Use this when the
   RL signal is weak or when you want to improve speculative decoding without
   affecting the policy.

3. **How are draft weights synced to vLLM?**
   Through vLLM's standard draft weight-update session: colocated engines use
   IPC and non-colocated engines use NCCL.

4. **What is `--dspark-block-size`?**
   The number of tokens the draft model predicts in parallel per block. Larger
   values increase potential speedup but may reduce acceptance rate. The
   default (7) works well for most models.

5. **How do I choose `--dspark-target-layer-ids`?**
   These are the policy layer indices from which hidden states are captured as
   input to the draft model. For a 36-layer model, `"1,9,17,25,33"` samples
   every 8th layer. More target layers = richer draft input but higher cost.

## References

1. [DSpark Paper](https://arxiv.org/abs/2505.14269) — Semi-autoregressive speculative decoding.
2. [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
3. [vime Speculative Decoding Docs](../../docs/en/advanced/speculative-decoding.md)
