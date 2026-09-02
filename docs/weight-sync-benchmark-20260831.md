# Disk weight-sync benchmark (4B and 30B)

## Scope

- Branch: `feature/disk-full-weight-sync`
- Transport: `disk`
- Modes: `delta` and `full`
- Rounds: 5 training updates. The initialization/baseline sync is excluded.
- Steady mean: arithmetic mean after excluding the first training update.
- 4B placement: 4 actor NPUs + 4 rollout NPUs.
- 30B placement: 8 actor NPUs + 4 rollout NPUs.
- Common workload: rollout batch 16, 4 samples/prompt, response length 2048,
  global batch 16, learning rate `1e-6`, vLLM TP 4, eager mode.

The earlier response-length-256 run was invalid: it produced all-zero rewards,
zero gradients and empty Delta manifests. These results use response length 2048,
contain non-zero gradients, and every Delta round has a non-zero payload.

## End-to-end update time

The total-time source is the outer `Timer update_weights` log. Values are in
seconds; the steady mean excludes round 1.

| Model | Mode | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Steady mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | Disk + Delta | 6.7 | 5.4 | 5.1 | 5.4 | 4.9 | 5.2 |
| Qwen3-4B | Disk + Full | 5.2 | 5.4 | 5.5 | 5.4 | 5.4 | 5.4 |
| Qwen3-30B-A3B | Disk + Delta | 82.7 | 60.8 | 61.5 | 59.2 | 117.1 | 74.7 |
| Qwen3-30B-A3B | Disk + Full | 67.7 | 52.3 | 62.5 | 55.0 | 56.6 | 56.6 |

At 4B the two modes are close: Delta's steady mean is about 4.1% lower than
Full's. At 30B, Full's steady mean is 24.2% lower than Delta's (equivalently,
Delta is 31.9% higher than Full). The 30B Delta mean includes a real fifth-round
outlier: reload rose to 46.8 seconds and total update time rose to 117.1 seconds.

## Delta payload

| Model | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B density | 1.47% | 1.43% | 1.04% | 1.03% | 0.97% |
| Qwen3-4B payload | 0.29 GB | 0.28 GB | 0.21 GB | 0.21 GB | 0.20 GB |
| Qwen3-30B-A3B density | 1.22% | 1.10% | 0.85% | 0.61% | 0.73% |
| Qwen3-30B-A3B payload | 1.85 GB | 1.69 GB | 1.35 GB | 1.01 GB | 1.18 GB |

## Stage breakdown and bottleneck

For 4B Delta, steady encode is about 1.9-2.2 seconds, materialization about
2.1 seconds, and reload about 0.7 seconds. Full writes the entire checkpoint in
about 4.5-4.8 seconds and reloads in about 0.6-0.7 seconds. Delta only has a
small advantage because it still scans/encodes the model and materializes a full
rollout-side checkpoint.

For 30B Delta, encode costs 25.9-41.6 seconds and rollout-side materialization
costs 20.1-30.9 seconds. Reload normally costs 6.0-11.1 seconds, but reached
46.8 seconds in round 5. The small 1.0-1.9 GB wire payload therefore does not
translate into lower end-to-end time: full-model scanning/encoding plus full
checkpoint materialization dominates.

For 30B Full, full checkpoint writing is stable at 45.1-48.6 seconds and is the
main cost. Reload varies from 5.9 to 20.7 seconds. Unlike Delta, Full avoids the
extra encode + materialize pipeline, so it is faster in this implementation even
though more bytes are written.

## Validation evidence and logs

All four jobs completed successfully. Representative non-zero gradients are
`0.660258` for 30B Full and `0.485568` for 30B Delta. The benchmark script also
rejects a completed job when no non-zero gradient exists, and rejects Delta when
any round reports `density=0.00% wire=0.00 GB`.

Original logs are retained on the remote host:

- `/home/vllm/weight-sync-bench/logs/real-20260830-4B-delta.log`
- `/home/vllm/weight-sync-bench/logs/real-20260830-4B-full.log`
- `/home/vllm/weight-sync-bench/logs/real-20260830-30B-delta.log`
- `/home/vllm/weight-sync-bench/logs/real-20260830-30B-full.log`

Artifact directories are under `/home/vllm/weight-sync-bench/real-20260830-*`.

## Reproduction

Use `scripts/run-weight-sync-benchmark-npu.sh` and set `MODEL_SIZE` to `4B` or
`30B`, and `UPDATE_WEIGHT_MODE` to `delta` or `full`. The script starts a clean
Ray runtime, binds NPUs 0-11, writes a durable log, and validates that training
actually mutated the model before accepting the run.
