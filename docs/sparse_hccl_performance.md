# Sparse HCCL: EP correctness and measured performance

## Scope and results

Qwen3-30B-A3B, one Ascend host, four training and four rollout NPUs,
TP=2, PP=2, EP=2, expert TP=1, BF16 eager execution, linear EP placement,
NZ mode off, CPU snapshots. Three rollouts, eight samples, response length 64.
These results are not a two-host or Qwen3-235B benchmark.

| Mode | Initial update (s) | Three steady updates (s) | Steady mean (s) |
| --- | ---: | --- | ---: |
| Full baseline | 114.0 | 43.4 / 47.1 / 40.1 | 43.5 |
| Previous sparse, verification enabled | 808.3* | 84.3 / 75.0 / 69.3 | 76.2 |
| Revised sparse, timing disabled, verification enabled | 680.8 | 29.2 / 26.6 / 27.1 | 27.6 |

Table note: seconds are rank-zero outer weight-update wall time, including
export, transport and receiver application/wait; ranks are not summed. Initial
update is excluded from the steady mean. Sparse uses checksum and local changed
entry verification every update; full has no additional sparse verification.
*The previous sparse cold update overlapped oracle work and is not a fair cold
comparison. Payloads vary between training runs; only three steady samples were
measured. This is an observed comparison, not a performance guarantee.

Revised steady latency is 36.5% below full and 63.7% below previous verified
sparse. Cold initialization remains substantially slower than full. Entire
three-rollout job duration, including startup/generation/training, was 1290.315 s
for revised sparse versus 875.465 s for full: the short end-to-end job is not faster.
Revised sparse transmitted 300.98 / 441.49 / 457.30 MiB in its steady updates.

## Interface flow and limited changes

Export/diff -> trainer gather/pack -> checksum -> HCCL -> receiver checksum ->
ownership mapping -> direct indexed write or native fallback -> local verification
-> completion/resume. Transport protocol, patch schema and checksum are retained.

- Receiver `sparse_weight_patch.py::_apply_direct_moe_patches` checks global
  expert IDs against the expert map, then checks mapped local slots against
  parameter storage. Nonlocal experts are skipped only with a valid static map.
  Dynamic EPLB goes to the native fallback before cached ownership filtering.
- Sender `sparse_gather.py::gather_slot_entries_to_rank0` preserves count exchange and
  merge order, but HCCL P2P sends actual peer lengths and skips empty peers.
  Other backends retain their padded path. Returned merged payload owns storage.
- Bucket/workspace objects are allocated only on cache misses.
- `--update-weight-stage-timing` is false by default. Per-update metadata enables
  receiver timing; disabled timing introduces no profiling synchronization.
  Enabled nested stage times are exclusive synchronized wall time, not GPU kernel
  time. See [stage timing](sparse_hccl_stage_timing.md).

## Where remaining cost comes from

A separate profiled run measured cold sender export 507.32 s, checksum 47.46 s,
HCCL completion 2.81 s and receiver wait 131.52 s. The prototype did not separately
instrument index construction/snapshot priming; missing fields are not zero.
Receiver checksum totaled 63.03–70.11 s per rank over 115 cold flushes. Do not sum
receiver ranks or add their overlapping times to sender times.

Steady profiled sender means were export 8.60 s, gather/pack 18.91 s, checksum
0.63 s, transfer 0.15 s and receiver wait 4.47 s. Gather/packing was the largest
measured steady stage. This profile preceded unpadded P2P and final diagnostic
refinements, so its 33.9 s mean versus 27.6 s cannot isolate timer overhead.

Reference inspection used committed verl revision
`5026915b60fb0a99a0d6e929b4b5af9a4ceae1f8`, not unrelated local modifications:

- `verl/checkpoint_engine/delta_sync/encode.py` uses the same underlying
  `torch.hash_tensor(...).item()` calls. Ascend logs report unsupported
  `aten::hash_tensor.out` falling back to CPU. Identical calls therefore do not
  imply identical backend performance; VIME also normalizes hash width for wire
  encoding. Integrity was not weakened to bypass this cost.
- `verl/workers/rollout/sglang_rollout/delta_loader.py` uses NaN masking and native
  loading, but does not provide the same Ascend EP mapping implementation.
- verl vLLM/SGLang rollout update paths use colocated device IPC where applicable;
  these are not equivalent to cross-process HCCL sparse updates. The closest
  network reference, `nccl_checkpoint_engine.py`, overlaps receives with persistent
  buffers. VIME retains per-flush completion/receiver waits; no buffering redesign
  was introduced.
- Full uses direct HF export, while sparse uses Bridge export plus snapshots.
  Cold export overhead cannot be attributed to HCCL alone. CPU snapshots were
  retained because a model-sized device mirror risks exhausting training memory.

No verl end-to-end speed comparison was run. Combined changes and host/cache
variation prevent assigning the whole observed improvement to the EP fix alone.

## Correctness and reproducibility evidence

223 regression tests passed, including real NPU EP/TP cases, global-to-local
expert mapping, changed and unchanged fixture tensor contents, default-off timer
behavior and dynamic EPLB fallback. A standalone four-NPU HCCL run passed 20
FP32/BF16 unequal/empty/noncontiguous/split/scratch-reuse cases. The optimized
end-to-end job succeeded with checksum and verify-every-1, timing disabled.
Fixture full-tensor comparison and changed-local-entry runtime verification do
not establish full-model 30B dense-oracle equivalence.

The benchmark used the revised static-EP implementation before the final dynamic
EPLB guard refinement; EPLB was disabled. That refinement passed the regression
suite and does not change the benchmark's static-EP branch.

| Run | Ray job | Raw log SHA256 |
| --- | --- | --- |
| Full | raysubmit_amLAaWuMVeUVGuqv | 62266af396737e072b6ec885ec2b0771abb6ee086a684976a572525dc6e21c40 |
| Profiled sparse | raysubmit_mi6ZESJ2wnK3YYL1 | 5a69e47ecec0d255d0f923d134927198ab032a18f8efdaf6fb0d0c3a58978425 |
| Revised sparse | raysubmit_BBied3rpth18LzpA | ff6ec6d1b731cd9e284210c094f662deabaeed319429ff82340c12f60716fe22 |

Table note: job IDs identify terminal SUCCEEDED runs with driver exit code zero;
SHA256 identifies complete raw logs, not model weights. Logs are retained with
the local validation artifacts. Reproduction must match topology, model/data,
verification and timing flags; changing sampled data changes sparse density.
