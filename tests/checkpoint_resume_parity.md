# Native Adam checkpoint resume and dense parity

The Megatron patch handles a fresh native Torch Adam/AdamW optimizer when the checkpoint loader requests a state template, and restores an independent step scalar for each parameter. A shared scalar is incremented once per parameter by Torch Adam, corrupting the next update. The patch also removes the checkpoint-only group step field after rebuilding per-parameter steps and keeps the restored step when loading parameter shards, which do not contain it.

`tests/utils/test_native_adam_checkpoint.py` exercises the real MCore methods using two CPU AdamW parameters: the original code fails the empty-state and step-independence checks; the patched code preserves the fifth update exactly and still rejects inconsistent saved steps or missing Adam moments. This test requires the Megatron version and patch installed by the container.

Existing checkpoint smoke tests demonstrate that loading runs. The integration test compares
the state and the next updates against uninterrupted training with identical saved
batches. It uses the real training loop, Megatron checkpoint loader, optimizer,
and optional `--offload-train` cycles. The audit hook only hashes detached tensors;
it does not replace training or restore any state itself.

Prepare a JSON array containing the arguments for a working single-GPU dense
Megatron training run, including `--num-rollout 8`, a fixed initial `--load`, and
`--load-debug-rollout-data /absolute/dumps/{rollout_id}.pt`. Keep the same eight
saved rollout files in all runs. They must retain tokens, masks, sampled logprobs,
rewards and grouping metadata. Use nonzero-gradient batches, zero dropout, and a
fixed runtime. Omit vLLM-only options because debug replay does not start serving.
This is a deterministic correctness test, not a timing benchmark.

Run from the repository root in three separate processes. The runner captures
stdout/stderr in `train.log` in each output directory. Each output must be new.

```bash
python -m tests.test_checkpoint_resume_parity record --train-args initial.json --output /tmp/parity-A
python -m tests.test_checkpoint_resume_parity record --train-args initial.json --output /tmp/parity-B0 --stop-after 4
# In resumed.json change only --load to /tmp/parity-B0/checkpoints.
# Keep --num-rollout 8 and omit --start-rollout-id, --finetune,
# --no-load-optim and --no-load-rng so the loader controls restoration.
python -m tests.test_checkpoint_resume_parity record --train-args resumed.json --output /tmp/parity-B1
python -m tests.test_checkpoint_resume_parity compare --continuous /tmp/parity-A --first /tmp/parity-B0 --resumed /tmp/parity-B1 --output /tmp/parity-result.json
```

The first branch runs all eight updates. The split prefix stops after update four
without changing the scheduler horizon. The resumed process must load the prefix
checkpoint and run updates five through eight. Verification requires separate
trainer PIDs, complete per-step evidence, nonzero updates, identical model and FP32
master parameter hashes, Adam moments/step, scheduler, RNG, gradients, loss metrics,
and training dump values (including selected-token forward logprobs and advantages).
The restored state before update five must also exactly match the prefix's final
state. Missing, duplicate or skipped evidence fails verification.

The current fixture is TP=PP=CP=1 and one update per rollout. It intentionally does
not claim topology resharding, full-vocabulary logits, stochastic dropout, dataset
cursor recovery from online generation, or live serving synchronization. Test the
first post-resume weight transfer and subsequent live generation separately. GPU
allocation and process cleanup belong to the calling CI/supervisor; never use
host-wide `pkill` or `ray stop` to run this test on shared machines.
