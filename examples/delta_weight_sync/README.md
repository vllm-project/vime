# Delta Weight Sync

VIME currently provides a direct delta-weight-update (DWU) MVP for a
non-colocated Megatron trainer and VIME-launched vLLM rollout engines. The
trainer exports canonical Hugging Face/checkpoint-coordinate tensors, keeps a
committed CPU snapshot, and sends:

- a dense BF16 seed for version 1; then
- absolute BF16 values plus flattened `int32` checkpoint indices for elements
  whose bit patterns changed after a committed optimizer step.

The vLLM worker applies those patches through the model's native
`load_weights()` mapping. Consequently, VIME does not need to know vLLM's
runtime QKV/gate-up packing or tensor-parallel parameter names.

```text
Megatron TP/DP weights
        -> canonical HF export
        -> dense seed or absolute sparse checkpoint patches
        -> NCCL
        -> vLLM CheckpointWeightPatch
        -> native model.load_weights()
```

## vLLM dependency

Direct DWU requires a vLLM build that contains the checkpoint weight patch
API (`CheckpointWeightPatch` and `load_checkpoint_weight_patches()` in
`vllm.model_executor.model_loader.checkpoint_weight_patch`) from
[vLLM PR #50723](https://github.com/vllm-project/vllm/pull/50723).

Until #50723 merges, no vLLM release contains that API; build vLLM from the
PR branch. This path was tested at PR commit
`fd07acd5b596c11f949fa71b5f0ee926b9e6bf17`; vime fails fast at engine startup
if the patch API is missing.

## Enabling direct DWU

Add to a non-colocated Megatron training run (vime starts and owns the vLLM
engines; do not set `--rollout-external`):

```bash
--update-weight-mode delta \
--update-weight-transport nccl
```

The first sync ships a mandatory dense seed (start version 0) that aligns the
rollout weights with the trainer checkpoint; every later sync ships only the
weights whose BF16 bits changed.

## Current MVP boundary

The direct path currently requires:

- `--train-backend megatron`;
- non-colocated, VIME-launched rollout engines;
- BF16, unquantized weights;
- Megatron PP=1 and VPP=1;
- vLLM PP=1 and DP=1;
- no rollout offload, speculative decoding, MTP draft update, fault-tolerant
  worker replacement, or fully-async rollout; and
- version 0 startup followed by a mandatory dense seed.

The source currently performs a full canonical-HF export and keeps the
committed weights in CPU memory. Steady-state network traffic is sparse, but
source traversal and snapshot memory are not yet sparse or sharded.

Delta over disk is only a reserved interface. The current argument validator
rejects:

```bash
--update-weight-mode delta --update-weight-transport disk
```

with `NotImplementedError`. The existing GLM disk script is retained as
historical interface material; it is not a runnable path for the current MVP.

## Verifying a run

A successful direct-DWU run must show all of the following in the Ray job
log:

1. Version 1 logs `dense_seed=True` with `changed == total`.
2. After a real optimizer step, version 2 or later logs
   `dense_seed=False`, `0 < changed < total`, and a smaller `wire_bytes` than
   the dense seed.
3. Training reports a finite, nonzero `train/grad_norm`.
4. Rollout generation succeeds after the sparse commit and no worker reports a
   base-version, sequence, final-manifest, or failed-session error.

The updater exports these step metrics after a committed update:

```text
weight_sync/is_dense_seed
weight_sync/total_elements
weight_sync/changed_elements
weight_sync/delta_density
weight_sync/wire_bytes
weight_sync/seconds
```

Its summary line has this form:

```text
Direct DWU committed version=<N> dense_seed=<bool> changed=<M>/<T> \
density=<ratio> wire_bytes=<bytes> seconds=<seconds>
```

Process launch, a dense seed alone, or static tests do not by themselves
demonstrate a working delta path; check all four criteria above.
