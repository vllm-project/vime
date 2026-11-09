# Huginn recurrent policy training

Train [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) at checkpoint
revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`. The provider preserves the
prelude, recurrent core, coda, sandwich norms and checkpoint's fixed
`mean_recurrence`. Each completion carries its own `like-init-cpu-f32-v1` latent
seed/position trace, replayed under current actor and critic weights. Its RoPE
buffer remains FP32 when parameters move to FP16.

| `--algorithm` | Objective | Critic | Defaults |
| --- | --- | --- | --- |
| `ppo` | Clipped rollout-policy ratio | Yes | GAE, one completion/prompt |
| `grpo` | Clipped group-relative advantages | No | Four completions/prompt |
| `dppo` | Decoupled PPO with proximal/behavior correction | Yes | Recompute proximal scores; correction clipped to [0.5, 2] |
| `flow-dppo` | Categorical full-vocabulary KL gate | Yes | Budget 0.01, two optimizer steps/rollout |

The [shared recipe](../looped_ppo) defines each objective, environment and
checkpoint contract. DPPO uses VIME's three-policy importance correction;
Flow-DPPO applies its divergence gate to categorical token probabilities.

Use the pinned native engine and patched Megatron from the shared recipe, with
one training GPU and one dedicated rollout GPU. The model defaults to FP16,
fixed loss scale 128 and FP32 gradient accumulation. JSONL rows contain string
`prompt` and `label` fields for the existing math reward.

```bash
python examples/huginn/run.py --algorithm grpo \
  --model /models/huginn-0125 --data /data/math.jsonl \
  --output /runs/huginn-grpo --updates 3 --recompute \
  --model-revision bb6621b65e90b6a4b9b29ef88dc83866d450470c \
  --engine-revision d2a358933393f25d74dd2bdd1068a741cd5d9226
```

Run each objective in a separate directory. For a fresh-process resume, add
`--stop-after 2` to an interrupted run, then rerun with `--resume` and without
that flag, keeping total updates and all settings fixed. Compare full
actor/critic, optimizer, scheduler and RNG state, tokens, rewards, latent traces
and publication digests with uninterrupted training. GRPO restores only its
actor; equal group rewards correctly produce zero advantages.

The tied embedding is published under both native aliases, checked for equality
by the engine. The fixed RoPE buffer must match the model revision. Complete
weight publication commits before generation resumes. Persist PyTorch version,
dtype and CPU architecture with the latent profile when comparing fresh runs.
