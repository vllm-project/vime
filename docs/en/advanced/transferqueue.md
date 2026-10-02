# Same-job trajectory transport with TransferQueue

The optional `simple-storage` mode streams complete text prompt groups to CPU SimpleStorage while other generation requests continue. It reuses the existing GRPO generation, reward/filter and batch conversion. Optimization still starts after a complete, fixed round is sealed; this first version does not overlap optimization with generation of the next policy.

The default `--transfer-queue-mode off` does not import TransferQueue or create its actors, client or heartbeat. Enabled mode requires the optional dependency `TransferQueue==0.1.9` (`transferqueue` package extra) in the runtime. It supports one text actor, TP=PP=CP=DP=1, no VPP, GRPO, the standard synchronous `train.py` and a checkpointable global dataset. Critic, partial rollout, fault recovery, debug replay, custom conversion/post-round transforms, multimodal data, MTP and MoE replay are rejected. Runtime codec checks unsupported Sample fields before writing them.

Add these flags to a working synchronous recipe:

```bash
--transfer-queue-mode simple-storage \
--transfer-queue-job-id training-001 \
--transfer-queue-restart-epoch 0 \
--transfer-queue-max-groups 64 \
--transfer-queue-max-tokens 131072 \
--transfer-queue-max-bytes 268435456 \
--transfer-queue-timeout-s 300 \
--transfer-queue-lease-s 120
```

Group capacity must contain the entire rollout batch. Token/byte limits reserve capacity before writing tensors; oversized groups or rounds fail rather than wait indefinitely. Byte accounting includes five tensor/transport buffers, owned JSON and decoded Sample fields. Only one codec/write/read runs at a time. The coordinator retains descriptors and checksums rather than a second payload copy. Revalidation at sealing processes one group at a time. Ray/allocator metadata and model activations are outside this working-set budget; record process RSS alongside `peak_working_bytes`. There is no GPU zero-copy path or process RSS guarantee.

## Identity, versions and consumption

Each complete group/attempt has a unique partition under its job and restart epoch. An initial put never appends to a shared partition. Original group/sample/rollout identities, rewards, masks, log probabilities and ragged top-p data survive encode→storage→decode. A nullable rollout ID remains nullable in Sample; the contract uses VIME's existing sample-index fallback, never the consumer's round number.

READY requires a successful real payload read and checksum validation. Exact duplicate publication is idempotent; conflicting or late groups are rejected using the committed-round watermark. Every child must carry the same actual policy version as the acknowledged serving cohort. Missing/mixed versions are rejected rather than filled from the driver.

After sealing, admission pauses, serving engines pause and the maintenance drain probe must confirm idle. UNKNOWN/BUSY or partial publication prevents further admission in this opt-in mode. Serving publication must acknowledge one changed version across the whole cohort before the next round.

The coordinator transitions READY→LEASED→PREPARED→TRAINING→TRAINED→COMMITTED. A separate heartbeat renews leases during the complete actor training plan. Reading metadata, conversion or one microbatch is not a training acknowledgment. Expired LEASED/PREPARED handles can be replaced with a fenced generation; expired/uncertain TRAINING/TRAINED becomes UNKNOWN and stops the job. No automatic optimizer replay or cross-crash exactly-once guarantee is provided.

Committed payloads are deleted before capacity is released. A deletion failure leaves pending GC and stops progress without undoing the optimizer or re-training the group. `reclaim_committed()` retries only owned deletion; it does not resume a failed training plan. Failed reads/writes pause admission and attempt cleanup of that partition. A timed-out remote write may finish late; its old epoch is never reused as training input.

## Checkpoints and validation

Checkpointing waits for a synchronous native save, data-source save, acknowledged publication and a quiescent queue. `transfer-queue.json` records the common next rollout cursor, sampling digest, job and committed watermark. Restart from that common checkpoint with the same job ID and a strictly larger restart epoch. Mismatched cursors/configuration fail before queue actors start. SimpleStorage is in-memory transport, not a durable optimizer transaction; incomplete model/data/queue snapshots cannot be resumed as a common checkpoint.

```bash
python -m pytest -q tests/rollout/test_transfer_queue_contract.py \
  tests/rollout/test_transfer_queue_adapter.py tests/test_megatron_argument_validation.py
python -m pytest -q tests/integration/test_transfer_queue_backend.py
```

The backend test uses real controller/storage/client actors for streaming, lease/commit, timeout cleanup, GC retry and a new-epoch restore. Its policy/training completion events are fixtures, so it is not GPU model E2E. Full E2E must run vLLM→TQ→Megatron→optimizer→next serving version, then fresh-process resume and compare against queue-off. Report matched repeated timings including `put_s`, `get_s`, payload bytes, peak working bytes, RSS and total round time. A generation/optimization barrier can add overhead; no speedup is promised.

Close only this job's client and let its owning processes exit normally. Do not use global TransferQueue close, stop shared Ray services, or delete another job's partitions.
