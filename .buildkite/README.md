# vime CI on Buildkite

Buildkite runs the always-on CPU jobs and the manual GPU-suite gate for vime.

The always-on steps live in [`pipeline.yml`](./pipeline.yml) and run on every
build (PR and push to `main`):

| Step | Purpose | Queue (machine) |
|---|---|---|
| `pre-commit` | pre-commit gate | `small_cpu_queue_premerge` (r6in.large) |
| `plugin-contracts` | plugin contract tests (19 files) | `medium_cpu_queue_premerge` (r6in.4xlarge) |
| `agent-adapter` | agent adapter tests (3 files) | `small_cpu_queue_premerge` |
| `utils` | utils tests (`pytest tests/utils`) | `medium_cpu_queue_premerge` |

The three test steps depend on the pre-commit gate. Each suite runs its files
sequentially inside one step because these queues boot a fresh EC2 instance
per job — a per-file matrix would be mostly boot + pip-install time. All
always-on CPU steps use the standard `python:3.11` image and install their
lightweight dependencies at runtime.

## Creating the pipeline (one-time, Buildkite UI)

Org `vllm`, cluster **CI** (the premerge queues live there).

1. New pipeline: name `vime-ci`, repository
   `https://github.com/vllm-project/vime.git`.
2. Leave the pipeline's Steps field as the default upload step — it reads the
   committed `.buildkite/pipeline.yml`:

   ```yaml
   steps:
     - command: buildkite-agent pipeline upload
       agents:
         queue: small_cpu_queue_premerge
   ```

3. GitHub settings on the pipeline:
   - Trigger builds after pushing code; branch filter: `main`.
   - Build pull requests (same-repository PRs only); skip builds for existing
     commits.
   - Update commit statuses.
   The Buildkite GitHub app must have access to `vllm-project/vime`.
4. Pipeline settings: enable **Skip Intermediate Builds** and
   **Cancel Intermediate Builds**.

No secrets are required for these steps (WANDB etc. is GPU-suite only).

## GPU suites

The GPU suites are behind a **block step** (`:rocket: Run GPU test suites?`):
click it in the Buildkite UI, multi-select the suites (`short`,
`vllm-config`, `megatron`, `precision`, `ckpt`), and the follow-up step
generates one job per test via [`gpu_suites.py`](./gpu_suites.py) — the same
`gpu_lock_exec.py` + `docker run` invocations used by the GPU jobs, including
the per-test `VIME_TEST_USE_DEEPEP` / `VIME_TEST_USE_FP8_ROLLOUT` /
`VIME_TEST_ENABLE_EVAL` combos.

The block uses `blocked_state: passed`, so a build whose CPU steps are green
reports a passing commit status even if nobody unblocks the GPU gate.

GPU jobs run on the shared **`mithril-h100-pool`** queue, following the same
pattern vllm-omni uses for it: each job is a Kubernetes pod (agent-stack-k8s
`kubernetes` plugin) on an H100 SXM node, with GPUs allocated via
`nvidia.com/gpu` limits (4 or 8), a memory-backed `/dev/shm`, and the node's
`/mnt/hf-cache` mounted as `HF_HOME`. vime tests `hf download` their models at
startup, so a warm HF cache is all they need. `WANDB_API_KEY` is not wired up
yet; runs report without wandb until it's added (e.g. as a k8s secret in the
pod spec).

## Keeping it in sync

The test lists live in `pipeline.yml` and `gpu_suites.py`; update both together
when adding or removing suites.
