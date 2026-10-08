# Sparse HCCL stage timing

Enable `--update-weight-stage-timing` for diagnostics. The
`run-qwen3-30B-A3B-sparse-hccl-tp2-pp2-ep2.sh` launcher accepts
`UPDATE_WEIGHT_STAGE_TIMING=1`. The default is off: no profiler is created,
no profiling device synchronization is issued, and no stage logs are emitted.
Checksum and the configured verification interval are unchanged.

Interface flow: trainer `update_weights()` creates an optional per-update
timer; `_publish_flush()` carries `profile` in the update metadata; each
rollout `receive_weights()` creates a per-flush timer and forwards it to the
sparse loader. There is no shared/global timer state.

`SPARSE-STAGES sender` records accumulated seconds on trainer rank zero for
one weight version. `SPARSE-STAGES receiver` records seconds on one rollout
communicator rank for one flush. Sum flushes **within a rank**, not across
parallel ranks, when comparing receiver totals. Sender transfer and receiver
transfer overlap; do not add them into an end-to-end latency.

Timing definitions (all values are synchronized wall-clock seconds, including
CPU work and completion of device work; not pure HCCL bandwidth):

| Stage | Boundary |
| --- | --- |
| sender export | Advance the dense HF or sparse diff iterator; excludes consuming yielded entries |
| sender gather_pack | Queue/gather sparse entries and consume into buckets; excludes nested checksum, transfer and receiver wait |
| sender checksum | Compute payload checksum, including any runtime CPU fallback |
| sender transfer | Send the already assembled HCCL buffers through completion |
| sender receiver_wait | Remaining RPC wait after the send, including receiver apply and scheduling |
| receiver transfer_allocate | Allocate receive buffers and receive through device completion |
| receiver checksum | Recompute and compare the negotiated payload checksum |
| receiver write | Sparse remapping, preparation and writing; excludes nested validation and verification |
| receiver validate | Validate sparse metadata/index bounds before writes |
| receiver verify | Read back changed local entries and compare to the received values |
| receiver verify_dense_replay | Optional dense replay/idempotence check, including replay writes |

Table note: initialization/export-index construction, snapshot priming,
generation pause/cache flush/resume and some outer packing are not represented
by these stages. Use the existing outer `update_weights` timer for total
latency. Stage boundaries explicitly synchronize when enabled, which changes
overlap and adds overhead. Measure performance with timing disabled as well;
profiled durations are attribution evidence, not a speedup claim.

EP regression: HF expert identifiers are global, while `w13_weight` and
`w2_weight` store local expert slots. Validate the global identifier against
`ascend_expert_map`, map it to a local slot, skip negative/nonlocal mappings,
then validate that slot against runtime storage. Do not compare the global
identifier to the local expert count. Existing layout, dtype, TP and EPLB
guards still retain the native-loader fallback.

Regression tests compare complete runtime tensors (both changed and unchanged
entries) for gate/up/down projections, both local slots, nonlocal and invalid
experts, CPU/NPU, float32/BF16 and TP1/TP2. These fixtures are not a proof of
full-model equivalence; retain the end-to-end rank-local verification and dense
oracle tests for their respective coverage.
