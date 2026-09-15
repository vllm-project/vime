# On-Policy Distillation Example

This example shows how to run **on-policy distillation (OPD)** using vime. A
small student (Qwen3-8B) is aligned to imitate a larger teacher (Qwen3-32B) by
training only on the student's own rollouts and matching the teacher's
token-level log-probabilities.

## Key Features

- **OPD is orthogonal to advantage estimators**: OPD works as an additive KL
  penalty on top of any advantage estimator (GRPO, PPO, REINFORCE++, etc.), not
  as a separate estimator.
- **Two teacher modes**:
  - **vllm**: Teacher runs on an external vLLM server; teacher log-probs are
    obtained during rollout.
  - **megatron**: Teacher is loaded directly into Megatron via
    `--opd-teacher-load`; teacher log-probs are computed during the training
    forward pass.
- **Student rollout always uses vLLM** (vime's default rollout backend).

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `vllm` or `megatron`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). |
| `--opd-teacher-load` | Path to teacher checkpoint. **Required** when `--opd-type=megatron`, **must not be set** when `--opd-type=vllm`. |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for teacher model. |

## Mode Comparison

| Mode | Teacher Location | When to use |
|------|------------------|-------------|
| `vllm` | External vLLM server | Teacher has different architecture or is larger than GPU memory |
| `megatron` | Loaded into Megatron training | Teacher has same architecture as policy/ref model |

## Components

- `vime/rollout/on_policy_distillation.py` implements (for vLLM mode):
  - `reward_func` calls the teacher server (via `args.rm_url`) with every sample
    to obtain token-level logprobs.
  - `post_process_rewards` trims the teacher logprobs to the generated response
    span and writes the tensors back to each `Sample` to compute advantages.
- `run-qwen3-8B-opd.sh` launches a vLLM teacher server, then submits a Ray job
  that runs `train.py`.
- `run-qwen3-8B-opd-megatron.sh` uses a Megatron-loaded teacher model (no
  external server needed).

## Running the example

### Using vLLM Teacher (External Server)

1. Download or prepare the required checkpoints and data.

```bash
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

2. Run the hf to mcore for student model conversion:

```bash
cd /root/vime
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```

3. Run on-policy distillation:

```bash
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

GPU layout:

| GPUs | Role |
|------|------|
| 0–3 | Student Megatron train + student vLLM rollout (colocate) |
| 4–7 | Teacher vLLM (Qwen3-32B, TP=4) |

### Using Megatron Teacher (No External Server)

1. Prepare student checkpoint (same as above).

2. **IMPORTANT**: Convert your teacher model to Megatron format (change the path
   to your actual teacher):

```bash
# This example uses the same model as both student and teacher (for demonstration only)
# In practice, use a different (stronger) model as the teacher!
cd /root/vime
source scripts/models/qwen3-8B.sh  # Or your teacher model config

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/YourTeacherModel \
    --save /root/YourTeacherModel_torch_dist
```

3. Edit `run-qwen3-8B-opd-megatron.sh` to update paths:
   - Change `--opd-teacher-load` to your teacher model path
   - Adjust `--opd-kl-coef` based on your task

4. Run:

```bash
bash examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
```

# Preliminary Results

End-to-end run with `run-qwen3-8B-opd.sh` (dapo-math-17k train, GRPO +
`--opd-kl-coef 1.0`, ~220 rollouts / iter_0000219). Offline GSM8K greedy eval:

| Model | GSM8K Accuracy |
|-------|----------------|
| Qwen3-8B (pre-OPD) | 79.7% (n=300) |
| Qwen3-8B (post-OPD) | **88.2%** (n=1319, **+8.5 pp**) |
| Qwen3-32B teacher | 87.0% (n=300) |

Training health signal: `rollout/opd_reverse_kl` dropped from 0.145 → ~0.10
(−38%). Pure OPD uses `raw_reward=0`; the learning signal is the OPD KL term.

# FAQ

1. **Why are there two OPD modes?**
   - `vllm` mode: The teacher runs on an independent vLLM server. This is useful
     when the teacher has a different architecture or is too large to load
     together with the policy model.
   - `megatron` mode: The teacher is loaded into Megatron using the same
     parameter loading mechanism as the reference model. This requires the
     teacher to have the same architecture as the policy model.

2. **How do I use Megatron-based teacher instead of vLLM server?**
   Replace your OPD arguments:
   ```bash
   # Instead of:
   --use-opd --opd-type vllm --opd-kl-coef 1.0
   # Use:
   --use-opd --opd-type megatron --opd-kl-coef 1.0 --opd-teacher-load /path/to/teacher_checkpoint
   ```

3. **What happens if I set wrong arguments?**
   The system will raise clear errors:
   - `--use-opd` without `--opd-type`: Error asking you to specify type
   - `--opd-type megatron` without `--opd-teacher-load`: Error asking for teacher checkpoint
   - `--opd-type vllm` with `--opd-teacher-load`: Error indicating conflict

4. **Why is `rollout/raw_reward` always 0?**
   Pure OPD distillation does not use an external reward model. The learning
   signal comes entirely from the OPD KL term applied to advantages.

5. **Self-distillation: why is `opd_reverse_kl` near 0 at the start?**
   Teacher and student start from the same weights, so reverse KL is ~0 until
   the student updates. For a true distillation signal, use a stronger /
   differently trained teacher (or `--opd-type vllm` with Qwen3-32B).

# References

1. https://thinkingmachines.ai/blog/on-policy-distillation/
2. https://arxiv.org/abs/2306.13649
3. https://arxiv.org/abs/2306.08543
