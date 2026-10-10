# CI (Continuous Integration)

Vime uses Buildkite for continuous integration. The committed pipeline is
`.buildkite/pipeline.yml`.

## Always-on checks

Every pull request runs these CPU steps:

| Step | Coverage |
|---|---|
| `pre-commit` | formatting, lint, and repository policy |
| `plugin-contracts` | customization contracts and CPU tests |
| `agent-adapter` | agent adapter behavior |
| `upstream-sync-cpu` | CPU tests synchronized from upstream |
| `utils` | `tests/utils` |

The authoritative commands and queue configuration are in
`.buildkite/pipeline.yml`.

## GPU suites

After the CPU steps pass, the Buildkite build exposes a block step named
`Run GPU test suites?`. Select one or more suites:

- `short`
- `vllm-config`
- `megatron`
- `vime-customized`
- `precision`
- `ckpt`

`.buildkite/gpu_suites.py` expands each selected suite into one Buildkite job
per test. Set `VIME_CI_IMAGE` to an immutable candidate digest when validating
Dockerfile or vLLM patch changes. Jobs otherwise use `vllm/vime:latest`, which
must not be updated before the change merges.

The `megatron` suite includes `test_straw_checkpoint_fork.py` for checkpoint
step selection, rollback branches and indexed debug archives.

`test_qwen2.5_0.5B_pipeline_rl.py` runs three GRPO steps with fully async rollout
on 4 GPUs. Its probes check that requests continue across weight updates and
that policy weights change. The matrix covers NCCL and full disk sync with
`--flush-cache-interval 0`, plus periodic refresh with NCCL and interval `2`.
`test_pipeline_rl.py` checks the schedule on CPU. These tests do not measure
learning quality or throughput gains.

### Coding-agent training

`tests/test_agent_sunabako_codex_e2e.py` uses Codex **0.162.1**, Claude Code
**2.1.296**, Qwen3.8-27B and two small MiMo tasks with separate pinned images:

- `format-code-task-000003`: implement smol-evm's missing `SHL` opcode.
- `format-code-task-000045`: preserve object quick replies in BootBot.

Each step samples **eight independent agents concurrently: four on each task**,
grades each patch in a fresh sandbox, and performs one GRPO optimizer step.
The first step uses Codex; the second uses Claude Code to sample both tasks again
with the updated model. Both CLIs use the same trace stitching and training path. There are exactly
sixteen agent runs and two training steps, without reward-based filtering or extra resampling.
Both unchanged repositories must fail their official tests before training.
Failed solutions retain reward 0 and participate in training alongside reward 1
solutions. GRPO normalizes each task's four outcomes separately: it subtracts the prompt-group mean and divides by its sample standard
deviation plus epsilon; a uniform group correctly has zero advantages.

The test compares the trainer's actual token advantages with those group statistics,
and records both optimizer calls on all eight training ranks. Gradients and
parameters must stay finite; nonzero gradients must produce parameter changes.
Sampling uses temperature 1, `top_p=0.95`, disabled top-k and score centering.
Inference enables the checkpoint's MTP head through vLLM's `mtp` method with
three speculative tokens and checks that draft tokens were actually accepted.
The 64K context budget bounds accepted tokens; vLLM manages lookahead slots internally. Claude Code uses
the example's six code tools: Bash, Read, Edit, Write, Glob and Grep.
Original token IDs, masks, sampler probabilities and nucleus replay data are
checked against the actual training tensors. Qwen3.8-27B is dense, so R3 is disabled.

Each agent has a 600-second budget. The Buildkite step timeout is 35 minutes
including setup. Continuous model turns
are merged using their original tokens and sampler distributions. Every segment of each
real trajectory enters training, as in the normal example.
`agents/<sample-index>/agent-full.pt` retains its complete trajectory.
The agent receives the original problem statement
and repository, while hidden tests stay in the independent grading sandbox.

Model files, the selected two task images, both CLI archives and TileLang/Triton
compilation results are cached. Interrupted image downloads resume and cached
blobs are verified. `--prepare-only` populates assets before GPU locks are acquired;
cold downloads can exceed the timed job limit. Proxy variables are propagated to
the container, with local Ray and sandbox traffic bypassing the proxy.
The Buildkite `megatron` suite includes this GPU test. The automatic
`agent-adapter` step covers the CPU contracts.

`tests/ci/setup_agent_e2e.sh` installs the released `sunabako==0.1.1` wheel from PyPI,
as pinned in `examples/coding_agent_rl/requirements-sunabako.txt`. It also prepares
the Ray authentication token before starting the head, reusing any existing token
and keeping its value out of CI logs. This sunabako release
supports native guest users with `uid_range_size`; the local node reserves 65,536 UIDs/GIDs for each of eight
sandboxes. State is mounted under `/workspace` so mapped users can traverse its
parent directories. This runs inside the privileged Buildkite test pod
without an inner Docker daemon or PRoot. The explicitly enabled RSS test mode is
a bounded functional check, **not aggregate hard RAM enforcement**. Production
sunabako still requires a writable delegated cgroup and fails closed.

For a preconfigured cluster, install the same requirements and run:

```bash
(umask 077; ray get-auth-token --generate >/dev/null)
HF_CHECKPOINT=/path/to/Qwen3.8-27B \
RAY_AUTH_MODE=token \
SUNABAKO_CLUSTER=/path/to/cluster.json \
SUNABAKO_IMAGES=/path/to/images.json \
ADAPTER_PUBLIC_HOST=<training-node-ip> \
python tests/test_agent_sunabako_codex_e2e.py
```

The image map must include both selected tasks on every sandbox node, with capacity for eight concurrent sandboxes. Set
`SUNABAKO_ALLOW_TEST_MEMORY=1` explicitly only when testing without hard cgroups.
Optional `VIME_AGENT_TEST_DATA` reuses the downloaded MiMo parquet/mapping;
`VIME_AGENT_CODEX_NATIVE_TARBALL` and `VIME_AGENT_CC_NATIVE_TARBALL` reuse the
official platform archives without downloading toolchains inside the sandboxes.
`VIME_AGENT_TEST_RUN_DIR` must name a new directory and retains the CLI logs,
grader output, rollout/train tensors, optimizer evidence and `result.json`.
The test uses the same `U.execute_train()` launcher as the other E2E tests, with
the agent environment passed through `extra_env_vars`. Ray's CLI prints training
logs directly to the Buildkite console. After training, the test archives
the job's log as `train.log` for the MTP and training-metric assertions.
Buildkite uploads the evidence on success and failure.

### Manual Megatron Restart

`test_qwen2.5_0.5B_training_recovery.py` uses 4 GPUs and two successive training jobs on the same Ray cluster. The first uses TP=1 and deliberately triggers a real CUDA OOM. After verifying that serving still responds when the job exits, it resubmits training with TP=2, which also changes the DP size.

The test checks that healthy vLLM processes, routers, and GPU placements are reused; replayed batch contents match; and training scheduler progress, finite nonzero gradients, and the final checkpoint are correct. The fixed test list includes:

| Data storage | RolloutManager state | Checks |
|---|---|---|
| straw with online GC | Remains alive | Reconnect trainers and replay completed training batches that were not checkpointed. |
| straw with online GC | Killed after failure | Reconnect a new manager to the original serving cluster and replay the same batches. |
| straw with a model/optimizer checkpoint and Megatron YAML configuration | Killed during training | Restore from the checkpoint and check configuration and recovery state. |
| Rollout debug files | Killed after failure | Restore data from debug files and reconnect a new manager to the original serving cluster. |
| straw with disk-delta weight synchronization | Killed after failure | Publish restored weights as a new full baseline, then continue delta updates. |
| straw with PD/NIXL serving | Killed after failure | Wedge the prefill actor, replace it within the reset timeout, and retain the healthy decode actor. |

`test_qwen3_30B_A3B_training_recovery.py` uses 8 GPUs for the same OOM/checkpoint/manager-loss workflow with a MoE model, R3, and stateless Adam. It omits optimizer tensors while checking scheduler progress, compares persisted routing bytes across the TP/DP change, and completes training after recovery. The dense cases cover ordinary Adam with optimizer checkpoints.

Internal serving health checks are enabled with or without the compatibility flag `--use-fault-tolerance`. CPU coverage includes configuration and checkpoint boundaries in `test_training_recovery.py`, lost disk-update replies in `test_disk_delta_recovery.py`, and real Ray manager SIGKILL or conversion-reply loss in `test_rollout_manager_recovery.py`.

### Removing Failed Engines at Rollout Completion

`test_qwen2.5_0.5B_rollout_health.py` stops a real vLLM HTTP server just before rollout completes, leaving its router registration intact. Two four-GPU cases either retain or kill the corresponding Ray actor. They check bounded rollout completion, deregistration before training, engine recovery at weight update, and a final training checkpoint.

Both cases omit `--use-fault-tolerance` and set the background interval and initial wait to 600 seconds. This verifies that rollout-completion checks run immediately without waiting for background checks.

## Registering tests

- Add always-on CPU tests to the appropriate command in
  `.buildkite/pipeline.yml`.
- Add GPU tests to a suite in `.buildkite/gpu_suites.py` and update the suite
  count shown by `.buildkite/pipeline.yml`.
- Keep `.buildkite/README.md` synchronized with pipeline behavior.

Run the exact command locally before triggering its remote Buildkite job. For
GPU failures, reproduce on an H200 node with the same image and environment,
then rerun the remote suite only after the local test passes.
