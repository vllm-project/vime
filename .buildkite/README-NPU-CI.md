# vime NPU CI on Buildkite

The NPU pipeline lives in [`pipeline-npu.yaml`](./pipeline-npu.yaml). It validates
vime on Ascend NPU hardware.

The NPU suites are behind a **block step** (`:rocket: Run NPU test suites?`):
click it in the Buildkite UI like GPU CI test suites.

## Pipeline steps

Three steps run in order:

**`pre-commit-npu`** runs the pre-commit gate on all files, always.

**`image-build-npu`** builds and pushes the NPU test image via buildctl/buildkit.
It only runs when `Dockerfile.npu` has changed (PR trigger) or on a scheduled
build — otherwise the step is skipped and the pre-built default image is used.

**`upload-npu-suites`** reads the `NPU_SUITES` environment variable (or
`buildkite-agent meta-data get npu-suites` for PR triggers) and generates
individual test jobs via [`npu_suites.py`](./npu_suites.py).

## Triggers

The pipeline supports three trigger modes:

**PR trigger.** On a PR trigger, the pre-built default image is used and
test suites are selected manually via the block step.

**Schedule trigger.** On a scheduled build, a new image is always built
regardless of file changes. The suites to run are determined by the
`NPU_SUITES` environment variable (set in the scheduled build's pipeline
configuration), which selects the corresponding entries from the `SUITES` dict.

## Adding a test

Suites and test mappings are defined in [`npu_suites.py`](./npu_suites.py). Two
suites are predefined — `smk` (always runs) and `nightly` (runs on schedule or
with the `run-ci-npu-nightly` label).

Each entry is a 4-tuple:

```python
("test-qwen3-4B-npu.py", "npu-8", "", {})
# (test_name, resource_class, extra_args, env_overrides)
```

- **`test_name`** — the test script under `tests/`, e.g. `test-qwen3-4B-npu.py`.
- **`resource_class`** — NPU count for the pod (`npu-2`, `npu-4`, `npu-8`,
  `npu-16`). 
- **`extra_args`** — extra CLI arguments passed to the test script.
- **`env_overrides`** — extra environment variables for the test step, e.g.
  `{"USE_DEEPEP": "1"}`.

A test script should:

1. **Download models and datasets to `HF_HOME`** before training so subsequent
   steps reuse the cache. The CI sets `HF_HOME=/root/.cache/huggingface`.

2. **Run training** via `train.py` or `train_async.py` with the appropriate
   NPU-specific arguments.

To register a new suite, add a key to the `SUITES` dict and update
`selected_suites()` if it should run conditionally. Also add a corresponding
option in the block step's `fields[].options` list in
[`pipeline-npu.yaml`](./pipeline-npu.yaml) so it can be selected in the
Buildkite UI.

To add a test to an existing suite, simply append a new entry tuple to its
list in `SUITES`.

## Adding or removing a patch

Patches live in `docker/npu_patch/` and are registered in the `PATCH_CONFIGS`
dict in [`update-npu-environment.sh`](./scripts/update-npu-environment.sh).
Update both when adding or removing a patch.

