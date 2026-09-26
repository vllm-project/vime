# Parameter-wise parallel replay diagnostics

A global gradient norm cannot detect a sign flip or a permutation. The optional
`--ci-save-parameter-grads DIRECTORY` diagnostic records every trainable dense
parameter after backward gradient finalization and before optimizer preparation,
unscaling or clipping. It leaves the existing gradient norm checks enabled.

Each rank writes a tensor-only file under
`DIRECTORY/ROLE/rollout-N/step-M/rank-R.pt`. Use a **fresh shared directory** for
each run. No extra collectives are added and live gradient buffers are not changed.
The diagnostic adds synchronous device-to-CPU copies and disk I/O; enable it only
for small-model correctness tests. The offline comparator loads the snapshots into
CPU memory, including reconstructed tensors for both runs.

The distributed optimizer's reduce-scatter leaves only part of each gradient
bucket valid on each DP rank. The collector saves that intersection, not the
whole `main_grad`. The comparator checks complete DP slice coverage, reconstructs
TP row/column partitions and gated MLP ordering, uses global PP layer names, and
checks overlapping replicas. Missing ranks, parameters, steps, nonfinite values,
and mismatched shapes fail explicitly. Numeric failures report the parameter,
logical tensor index, values, maximum absolute error and contributing ranks.

## Existing Qwen3 replay test (8 GPUs)

```bash
VIME_TEST_CHECK_PARAMETER_GRADS=1 \
VIME_TEST_GRAD_RTOL=0.01 VIME_TEST_GRAD_ATOL=1e-6 \
python tests/test_qwen3_0.6B_parallel_check.py
```

This retains one rollout replay for both loss reductions and adds gradient files
for the baseline and each existing layout. Temporary `parallel-grads-*`
directories are retained for debugging. Tolerances are elementwise
`abs(actual-reference) <= atol + rtol * abs(reference)`; configure them for the
precision and backend being tested. Defaults are starting values, not a claim
that every BF16 kernel/layout agrees at that tolerance. The full eight-GPU replay
must be validated in its training image before enabling this diagnostic by
default in GPU CI.

Compare saved runs again without launching training:

```bash
python -m vime.backends.megatron_utils.gradient_check /path/to/reference /path/to/replay \
  --rtol 0.01 --atol 1e-6
```

## Small real-Megatron smoke test (1–2 GPUs)

Use the Megatron revision pinned in `docker/Dockerfile`. The smoke test uses a
four-layer GPT (hidden size 64, vocabulary 128, grouped-query attention and gated
MLP), identical logical weights and tokens, unequal masks, and four accumulated
samples. It uses Megatron's real pipeline schedule, distributed gradient buffers,
NCCL and distributed Adam, with the local PyTorch attention backend. No model or
dataset download is required. Run from the repository with `PYTHONPATH=.` (plus
your Megatron checkout, if it is not installed).

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tests/gradient_check_megatron.py --out /tmp/grad-base
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/gradient_check_megatron.py --tp 2 --out /tmp/grad-tp2
python -m vime.backends.megatron_utils.gradient_check /tmp/grad-base/grads /tmp/grad-tp2/grads \
  --rtol 1e-3 --atol 1e-6
```

Omit `--tp 2` for DP=2, or replace it with `--pp 2` for PP=2. `--overlap` exercises
multiple gradient buckets; `--per-token-loss` exercises token normalization;
`--bf16` selects BF16. Always compare runs with the same precision and loss mode.
The smoke test also saves selected-token logprobs and post-Adam weights as
experiment artifacts and verifies that a repeated snapshot leaves gradient
buffers unchanged. This is a diagnostic integration smoke test, not a full RL
rollout/convergence test. BF16 TP may require larger absolute gradient tolerance
because the order of accumulation changes; do not infer optimizer-update parity
from gradient parity alone.

## Scope

Supported: dense BF16/FP32 Megatron DDP with untied embeddings/output weights and
one distributed optimizer instance. FP16 loss scaling, FP8, MoE/EP, tied weights,
and multiple distributed optimizer instances are rejected. CP replicas use the
DDP buffer's DP-with-CP group, but the local-attention smoke test does not exercise
SP or CP attention. Validate SP/CP through the full replay test in the standard image.
Post-update checks and selected-token logprob checks are not new production
flags in this patch.

CPU regression tests require PyTorch and pytest, without Megatron:

```bash
python -m pytest tests/test_gradient_check.py
```
