# Nanbeige recurrent policy training

See [verification results](RESULTS.md) for the tested source, retained evidence and remaining official-weight GPU gates.

Train [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) at checkpoint
revision `b82e54bd609793562a75cbf9337970a93369eab5`. The provider preserves independent
attention histories for each loop, the checkpoint's `head_dim`, and its loop-final
normalization. Shared physical parameters are optimized once and exported under
their native names; the PPO value head is kept out of policy publications.

| `--algorithm` | Policy objective | Critic | Defaults |
| --- | --- | --- | --- |
| `ppo` | Clipped ratio against rollout likelihoods | Yes | GAE, one completion/prompt |
| `grpo` | Clipped group-relative advantages | No | Four completions/prompt |
| `dppo` | Decoupled PPO: proximal/behavior importance correction | Yes | Recompute proximal scores; correction clipped to [0.5, 2] |
| `flow-dppo` | Categorical divergence gate with full-vocabulary KL | Yes | KL budget 0.01, two optimizer steps/rollout |

DPPO reuses VIME's three-policy training path described in
[Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641).
Flow-DPPO is a categorical extension of the
[flow-model algorithm](https://arxiv.org/abs/2606.11025); Nanbeige has a token policy,
so this computes full-vocabulary categorical KL rather than a Gaussian transition KL.

Install the [pinned native engine](../../vime/backends/vllm_rlt_utils/README.md)
and use Megatron `1dcf0dafa884ad52ffb243625717a3471643e087` with
[VIME's patch](../../docker/patch/latest/megatron.patch) in an isolated environment.
The Ray recipe needs one training GPU and one dedicated rollout GPU. JSONL rows
contain string `prompt` and `label` fields, graded by VIME's existing math reward.
Nanbeige defaults to FP32: its checked FP16 replay exceeded the fixed 0.03
selected-logprob error bound. Keep rollout and training precision identical.

```bash
python examples/nanbeige/run.py --algorithm grpo \
  --model /models/Nanbeige4.2-3B --data /data/math.jsonl \
  --output /runs/nanbeige-grpo --updates 3 --recompute \
  --train-env-vars '{"PYTORCH_ALLOC_CONF":"max_split_size_mb:128"}' \
  --model-revision b82e54bd609793562a75cbf9337970a93369eab5 \
  --engine-revision d2a358933393f25d74dd2bdd1068a741cd5d9226
```

The checked 40 GB A100 PPO run limits large allocator-block splitting with
`train-env-vars`; keep this setting when resuming.

Select each algorithm in a separate output directory. For an interrupted run,
add `--stop-after 2`, then start a fresh process with `--resume` and without that
flag. Keep the total update count, data, precision and batch settings identical.
Compare tokens, rewards, traces, actor/critic parameters, optimizer, scheduler,
and RNG state against uninterrupted execution. GRPO creates only the actor
checkpoint; all-equal rewards correctly produce zero group advantages.

Physical weight publications commit before the next rollout and record their
version and content digest. See the [shared recipe](../looped_ppo) for packing,
optimizer, cache sizing and timing conventions.
