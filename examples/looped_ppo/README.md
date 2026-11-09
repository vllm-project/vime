# Fixed-depth recurrent policy training

The shared launcher runs VIME's standard Ray entry with native vLLM-RLT rollout.
[Nanbeige](../nanbeige) documents its model-specific command and all four algorithms.
[Huginn](../huginn) uses the same four objectives and replay contract.
Ouro also uses this entry with `--model /models/Ouro-1.4B` and checkpoint revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

Use `--algorithm ppo`, `grpo`, `dppo`, or `flow-dppo`. PPO and GRPO reuse the existing
clipped policy loss. DPPO keeps rollout behavior scores separate from proximal
scores recomputed by the training provider, and applies VIME's existing bounded
importance correction. Flow-DPPO freezes full-vocabulary old-policy scores and
applies the categorical divergence gate. Actor/critic roles use GAE for PPO,
DPPO and Flow-DPPO; GRPO uses an actor and prompt-group reward normalization.

Install the [pinned RLT engine](../../vime/backends/vllm_rlt_utils/README.md) and
Megatron `1dcf0dafa884ad52ffb243625717a3471643e087` with
[VIME's patch](../../docker/patch/latest/megatron.patch) in an isolated environment.
Use an existing two-GPU Ray cluster; `--ray-address host:port` selects its address.
The launcher propagates its Python executable and source path to workers.
Math JSONL rows need string `prompt` and `label` fields, for example:

```json
{"prompt":"Calculate 7-4. Reply with only \\boxed{answer}.\n###Response\n","label":"3"}
```

Defaults use SGD without momentum, FP32 gradient accumulation, four prompts,
128 prompt tokens, 48 response tokens and full-vocabulary sampling. Nanbeige
and Ouro use FP32; Huginn uses FP16. `--precision fp16` selects fixed loss scale
128. Local MCore checkpointing
restores optimizer parameter groups, and FP16 master parameters and loss scale.
`--recompute` checkpoints each physical block. `--rlt-cuda-graphs` selects the
engine's graph path; eager execution is the default. Cache capacity accounts
for all recurrence planes of two active sequences.

`--rollout-batch-size` counts prompts and `--n-samples-per-prompt` counts completions.
GRPO defaults to four completions and requires at least two. Its reward normalizer
runs before samples are packed; a second batch-wide advantage whitening is omitted.
PPO/DPPO use one optimizer step per rollout. Flow-DPPO uses two steps, each with
half the completion batch. Extra arguments pass through to VIME.

Use a separate output directory with `--stop-after 2`, then a fresh process with
`--resume` for interrupted/resumed comparisons against three uninterrupted updates.
Actor and critic checkpoints are separate; GRPO saves only the actor. The dataset
cursor and policy version continue, and restored weights publish before generation.
Compare full model/optimizer/scheduler/RNG state, tokens, rewards and traces.
The launcher enables deterministic training and passes the cuBLAS workspace and
NCCL algorithm settings to Ray workers for this comparison.
Report end-to-end timing including rollout, rewards, optimization, full publications
and checkpoints, with source pins, dtype and resource contention.
