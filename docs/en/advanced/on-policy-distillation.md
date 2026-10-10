# On-Policy Distillation

On-policy distillation (OPD) trains a student on response tokens sampled from the student's current policy. At every visited prefix, a fixed teacher scores the same next token, providing a dense token-level learning signal along the student's own trajectories. In vime, this signal is a sampled reverse-KL penalty applied to the advantage, so it can be combined with an advantage estimator such as GRPO, PPO, or REINFORCE++. With zero task reward, the same mechanism performs pure distillation.

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--use-opd` | Enable on-policy distillation. Required flag to use OPD. |
| `--opd-type` | Type of OPD: `vllm` or `megatron`. Required when `--use-opd` is set. |
| `--opd-kl-coef` | OPD KL penalty coefficient (default: 1.0). Controls the weight of the distillation signal relative to the RL advantage. |
| `--opd-teacher-load` | Path to teacher Megatron checkpoint. **Required** when `--opd-type=megatron`, **must not be set** when `--opd-type=vllm`. |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for teacher model. |
| `--opd-teacher-model` | Optional served model name sent to the external vLLM teacher when `--opd-type=vllm`. |
| `--opd-teacher-scoring` | Text scoring path for a vLLM teacher: `prompt-logprobs` (default) or `per-position`. Image samples retain the default path. |

## How It Works

Let $\pi_\theta$ denote the student, $\pi_T$ the teacher, and $h_t$ the history before token $a_t$ on a student-generated trajectory. Following [Thinking Machines Lab's definition](https://thinkingmachines.ai/blog/on-policy-distillation/), the per-token reverse KL is

$$
D_{\mathrm{KL}}\left(\pi_\theta(\cdot \mid h_t) \| \pi_T(\cdot \mid h_t)\right)
= \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid h_t)}\left[
\log \pi_\theta(a_t \mid h_t) - \log \pi_T(a_t \mid h_t)
\right].
$$

The order is important: the student is the first argument of the KL, and the expectation is also over the student distribution. The teacher does not generate the training trajectory; it evaluates the token that the student actually sampled.

vime does not enumerate the full vocabulary to compute this expectation. For each sampled token, it uses the Monte Carlo contribution

$$
\hat d_t = \log \pi_\theta(a_t \mid h_t) - \log \pi_T(a_t \mid h_t),
\qquad a_t \sim \pi_\theta(\cdot \mid h_t),
$$

and modifies the base advantage as

$$
\hat A_t = A_t - \lambda_{\mathrm{opd}}\hat d_t.
$$

Here, $A_t$ is the advantage from the configured estimator (or zero for pure distillation), and $\lambda_{\mathrm{opd}}$ is `--opd-kl-coef`. An individual $\hat d_t$ may be negative even though the KL is non-negative in expectation. The policy loss uses $\hat A_t$, which makes the OPD term orthogonal to the choice of GRPO, PPO, REINFORCE++, GSPO, or another supported advantage estimator.

## Two Teacher Modes

### vLLM Mode (`--opd-type vllm`)

The teacher runs on an external vLLM server. Teacher log-probs are obtained during the rollout phase.

**When to use**: The teacher has a different architecture from the student, or the teacher is too large to load alongside the training model. Because the teacher scores the student's exact token IDs, the teacher and student must still use compatible tokenization and vocabularies.

**How it works**:
1. An external vLLM server runs the teacher model.
2. During rollout, the custom reward function (`vime.rollout.on_policy_distillation.reward_func`) sends the student's sampled token IDs to the teacher server and obtains the teacher log-probability of those same tokens.
3. The custom post-processing function (`vime.rollout.on_policy_distillation.post_process_rewards`) trims the teacher log-probs to the response span and stores them in `sample.teacher_log_probs`.
4. During training, vime subtracts the sampled log-probability difference, scaled by `--opd-kl-coef`, from the base advantage.

**Configuration**:
```bash
--use-opd
--opd-type vllm
--opd-kl-coef 1.0
--custom-rm-path vime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path vime.rollout.on_policy_distillation.post_process_rewards
--rm-url http://<TEACHER_IP>:<TEACHER_PORT>/inference/v1/generate
```

#### Optional per-position text scoring

Add `--opd-teacher-scoring per-position` to request only the actual response token's score at each position. VIME still sends the full prompt and response; each score keeps its full-vocabulary log-probability meaning. The loss, rollout temperature and learner input format are unchanged. Empty responses skip the teacher request and produce an empty score vector.

The teacher needs vLLM's [per-row candidate scoring](https://github.com/vllm-project/vllm/pull/56984), the V2 model runner, and the non-streaming `/inference/v1/generate` endpoint. For example, start a compatible teacher with:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-scale-out --logprobs-mode raw_logprobs
```

Do not use a logits output mode or `--kv-sharing-fast-prefill`. Candidate scoring skips prefix-cache reads to compute all requested scores. The existing `max_tokens=1` remains because the teacher interface still requires generation. Missing, malformed or invalid candidate scores raise an error rather than silently switching text scoring modes. Image samples continue to use the existing render, feature alignment and `prompt_logprobs` path, even when this option is enabled.

The default remains `prompt-logprobs` for compatibility with older teachers. Smaller score responses do not guarantee faster training, and short inputs may see no benefit.

### Megatron Mode (`--opd-type megatron`)

The teacher model is loaded directly into Megatron via `--opd-teacher-load`. Teacher log-probs are computed during the training forward pass.

**When to use**: The teacher has the same architecture as the student/reference model and fits in GPU memory.

**How it works**:
1. The teacher model is loaded as an additional Megatron model during initialization.
2. During the training forward pass, the teacher model computes log-probs for each sample.
3. The KL penalty is computed inline and applied to advantages.

**Configuration**:
```bash
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/teacher_torch_dist
```

> **Note**: The teacher checkpoint must be in Megatron format (`torch_dist` or `torch`). You can convert from HuggingFace format using `tools/convert_hf_to_torch_dist.py`.

## Running the Examples

Complete example scripts are provided in `examples/on_policy_distillation/`:

### vLLM Teacher

```bash
# 1. Download models and data
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k

# 2. Convert student model
cd /root/vime
source scripts/models/qwen3-8B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist

# 3. Run
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```

### Megatron Teacher

```bash
# 1. Convert both student and teacher models to Megatron format
# 2. Run
bash examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
```

## Preliminary Results

Using Qwen3-8B-Base model SFT-ed on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, on-policy distillation with a Qwen3-32B teacher on the remaining data yields:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |
