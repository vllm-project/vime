# vime NPU CI on Buildkite

The NPU pipeline lives in [`pipeline-npu.yaml`](./pipeline-npu.yaml). It validates
vime on Ascend NPU hardware.

The NPU suites are behind a **block step** (`:rocket: Run NPU test suites?`):
click it in the Buildkite UI like GPU CI test suites.

## Pipeline steps

Three steps run in order:

**`pre-commit-npu`** runs the pre-commit gate on all files, always.

**`npu-gate`** is a block step that pauses the pipeline for manual interaction.
It appears only on PR triggers (`build.source != "schedule"`).  You select which
NPU suites to run via a multi-select field:

- **`smk`** — runs all registered smoke cases on their declared hardware. Qwen3-4B
  runs twice, once on eight A3 devices and once on eight A5 devices (four training,
  four rollout). The other smoke cases run on A3. The uploader resolves
  `quay.io/ascend/vime:vime-latest` for A3 and `quay.io/ascend/vime:vime-latest-a5`
  for A5 once per build. Both tags must exist for a prebuilt smoke run.
- **`image-build`** — includes `smk` and builds a commit image for each hardware
  variant used by the selected tests. This mode can run before prebuilt tags exist.
- **`nightly`** — runs the nightly test suite.

**Image builds.** The uploader uses PyYAML to read the shared step from
[`pipeline-npu-image.yaml`](./pipeline-npu-image.yaml) for each requested hardware variant.
`image-build-npu` publishes `vime-ci-npu:<commit>` for A3.
`image-build-npu-a5` publishes `vime-ci-npu:<commit>-a5` using the matching
A5 base image and SoC. When building images, each test depends on its hardware's
build step. Prebuilt runs submit only the selected tests.

This allows testing code changes that require an updated NPU environment
(e.g., modifications to `docker/Dockerfile.npu`) before they are merged.

**`upload-npu-suites`** reads the `NPU_SUITES` environment variable (or
`buildkite-agent meta-data get npu-suites` for PR triggers) and generates
the required image builds and test jobs in one upload via [`npu_suites.py`](./npu_suites.py).

## Triggers

The pipeline supports two trigger modes:

**PR trigger.** The `npu-gate` block step appears for manual suite selection.
Suites not selected in the block step are skipped. The `image-build-npu` step
builds a new image only when `image-build` is chosen in the block.

**Schedule trigger.** There is no block step — the `npu-gate` step is omitted
entirely. A new image is **always** built, and the suites are determined by the
`NPU_SUITES` environment variable (set in the scheduled build's pipeline
configuration), which lists suite names from the `SUITES` dict.

## Adding a test

Each case in [`npu_suites.py`](./npu_suites.py) names its script and hardware
preset. A preset pairs a hardware variant with a Buildkite resource class.
Register the same script twice to cover two configurations:

```python
Case("test_qwen3_4B_npu.py", A3_8),
Case("test_qwen3_4B_npu.py", A5_8),
```

A case can also specify `extra_args`, `env`, or `timeout_in_minutes`.
The job label includes its hardware variant and resource class.

A test script should:

1. **Download models and datasets to `HF_HOME`** before training so subsequent
   steps reuse the cache. The CI sets `HF_HOME=/root/.cache/huggingface`.

2. **Run training** via `train.py` or `train_async.py` with the appropriate
   NPU-specific arguments.

For an existing hardware variant, reuse a preset or declare one for a different
card count. For a new variant, add its queue, image tags and optional SoC or
environment overrides to `HARDWARE`, then use its preset in the relevant cases.
The Buildkite selector lists suites and stays the same as hardware coverage grows.

With PyYAML installed, preview the generated build keys, dependencies and images before uploading:

```bash
NPU_SUITES=image-build BUILDKITE_COMMIT=$(git rev-parse HEAD) python3 .buildkite/npu_suites.py --pipeline
```

## Adding or removing a patch

Patches are applied explicitly by [`docker/Dockerfile.npu`](../docker/Dockerfile.npu)
and reconciled at runtime in the order declared by
[`docker/npu_patch/series.conf`](../docker/npu_patch/series.conf).
Each entry declares `target_worktree|image_patch|source_patch`: the image keeps
the flat `image_patch` under `/opt/npu_patch`, while runtime CI reads
`source_patch` relative to the current VIME checkout.

When adding a patch file, add the matching Dockerfile COPY/apply operations and
the `series.conf` entry in the same order. When deleting a patch, remove the
Dockerfile operation and `series.conf` entry before deleting the patch file;
the previous image retains the OLD bytes required for runtime revert. Rename
and reorder changes must likewise update both declarations. Ordinary patch
changes must not add patch-specific branches to the CI script.

Patch-only changes reuse the default image and are reconciled when the test
container starts. Changes that add, remove, or upgrade external repositories,
modify installed dependencies, or otherwise change the Docker image must be
validated with `image-build`.

`torch-memory-saver.patch` is applied only while building the TMS wheel in
`Dockerfile.npu`. It is not a runtime Git-source patch and must not be added to
`series.conf`: the TMS source tree is removed after installation. Changes to
this patch require `image-build` so Python and the allocator libraries match.

The initial rollout of this mechanism requires publishing a new default image
that contains `/opt/npu_patch/series.conf`, then updating
the corresponding `HARDWARE` entry in `npu_suites.py`. The legacy default image contains patch
bytes under `/tmp/npu_patch` but not the OLD series required by this reconciler.
