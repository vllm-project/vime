# Native sparse HCCL integration

This proposal is based on VIME #447 and reuses
[vllm-ascend #18915](https://github.com/vllm-project/vllm-ascend/pull/18915), pinned
to `988f993c1ab125372596dfcba9ba844832292453`.

## API and update flow

VIME's existing Megatron diff and TP/PP/EP-aware gather produce checkpoint names,
full shapes, int32 flat indices and replacement values. VIME constructs the public
`SparseWeightPatch`, initializes `SparseHCCLTrainerInitInfo`, and sends bounded
lists through `SparseHCCLTrainerWeightTransferEngine.send_weight_chunk` inside
its existing pause/start/update/finish/resume lifecycle.

The Ascend patch takes `sparse_hccl_engine.py` and `sparse_weight_patch.py` directly
from #18915. Every inference rank consumes all broadcasts before upstream
`is_pp_missing_parameter` filters parameters for its PP stage. The public vLLM
checkpoint loader performs TP/expert mapping. The temporary checkpoint loader
view restores Ascend's inference orientation in `finally` without replacing
runtime parameter storage. There is no direct-write or packed-wire extension.

The image's existing vLLM revision already provides the common sparse trainer
and checkpoint-patch loader, so no VIME-specific sparse extension is added to
`vllm.patch`. Existing non-sparse image compatibility patches remain.
The older Ascend image also needs the upstream communicator `close()` method
backported for native sparse trainer shutdown.

## Initialization and validation limits

The native sparse protocol has no values-only dense seed. The first update
therefore exports all HF weights and sends complete index/value patches, bounded
by `--update-weight-buffer-size`. For BF16 this uses six payload bytes per weight,
compared with two for dense HCCL; seed must be reported separately. Checkpoint
indices are absolute int32 coordinates, including across split pieces.

The native protocol has no checksum or receiver-local verification metadata.
`--update-weight-delta-verify-every` must be zero; nonzero values fail explicitly
rather than silently disabling a requested check. Correctness is validated by
the independent receiver E2E oracle and trainer-side full/sparse reconstruction.
Transport errors and potentially partial updates keep the existing failed-state
handling. This is not transactional rollback or an integrity checksum guarantee.

Supported scope remains unquantized ND weights with static expert placement,
EPLB disabled, and separate training/inference devices. MoE direct writes,
quantized/NZ/split expert weights and other model-family claims are excluded.

## Patches and validation status

The default `docker/Dockerfile.npu` dependency pins remain unchanged:
vLLM `e6bfe03ad73a3330cb427885aa90d97a12e1c704` and Ascend
`fd815467c221ee600137f6bdd53fe354d5e7c999`.

Both dependency patches pass pristine-source application checks and reverse-apply checks
against the patched remote trees. Container imports and native sparse HCCL
hardware correctness have passed. The native receiver TP2/PP2/EP2 test matches
complete worker parameter hashes and generated tokens against dense loading.
See [the functional logs and timing definitions](validation/pr18915/README.md).

VIME full-versus-sparse 30B end-to-end measurements are still **pending**; the full
baseline has started. Results from the earlier direct-write implementation do
not validate this candidate. Keep the PR in draft until the comparison is done.
The comparison uses the same container, model, TP2/PP2/EP2 layout, batch settings
and three steady training rounds. Whole update timers include diff/export,
gather, transfer, apply and finish; seed is reported separately.
