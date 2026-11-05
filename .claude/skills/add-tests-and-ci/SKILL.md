---
name: add-tests-and-ci
description: Guide for adding or updating vime tests and CI wiring. Use when tasks require new test cases, CI registration, test matrix updates, or workflow template changes.
---

# Add Tests and CI

Add reliable tests and integrate them with vime CI flow.

## When to Use

Use this skill when:

- User asks to add tests for new behavior
- User asks to fix or update existing tests in `tests/`
- User asks to update CI workflow behavior
- User asks how to run targeted checks before PR

## Step-by-Step Guide

### Step 1: Pick the Right Test Pattern

- Follow existing naming: `tests/test_<feature>.py`
- Start from nearest existing test file for your model/path
- Keep test scope small and behavior-focused

### Step 2: Keep CI Compatibility

- CI executes registered test files with `python tests/<file>.py`, not only pytest discovery. New CPU pytest files should include:

```python
import pytest

NUM_GPUS = 0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
```

- Set `NUM_GPUS = 0` for CPU-only tests, following the existing test metadata convention.
- For GPU/e2e tests, follow the nearby file pattern (`prepare()`, `execute()`, `NUM_GPUS`, and any model/dataset constants).

### Step 3: Register Tests in Buildkite CI

Whenever adding, moving, or renaming a test file, update its Buildkite registration before finishing:

1. Register CPU test files in the appropriate command list in `.buildkite/pipeline.yml`, beside similar tests. Agent CPU tests belong in `agent-adapter`.
2. Register GPU/e2e tests in `.buildkite/gpu_suites.py`, with the matching GPU count and environment settings. Update `.buildkite/pipeline.yml` when changing suite selection or wiring.
3. Include the registration changes with the tests. These files are the source of truth; there is no GitHub workflow regeneration step.

Only skip fixed matrix registration when the test is intentionally helper-only or manually invoked; state that reason in the final response.

### Step 4: Run Local Validation

- Run the exact existing test files you changed, if any.
- For new registered tests, run the same shape CI will use, for example `python tests/test_new_file.py`.
- Run repository-wide checks only when they are already part of the task or workflow.
- Avoid documenting placeholder test commands that may not exist in the current tree.

### Step 5: Keep Buildkite Sources in Sync

For CI workflow changes unrelated to a new, moved, or renamed test:

1. Edit `.buildkite/pipeline.yml` for always-on CPU commands and pipeline wiring.
2. Edit `.buildkite/gpu_suites.py` for generated GPU jobs rather than editing its generated output.
3. Keep suite definitions, selection, and `.buildkite/README.md` consistent when changing suites.

### Step 6: Provide Verifiable PR Notes

Include:

- Which tests were added/changed
- Where each new/renamed test was registered in `.buildkite/pipeline.yml` or `.buildkite/gpu_suites.py`
- Exact commands executed
- GPU assumptions for each test path
- Why this coverage protects against regression

## Common Mistakes

- Editing generated GPU jobs instead of their source
- Relying on pytest discovery for a new test in a suite with an explicit file list
- Treating a green CPU build as GPU validation; GPU suites require the manual Buildkite gate
- Adding a CPU pytest file that passes under `pytest tests/foo.py` but fails under CI's `python tests/foo.py`
- Adding tests without following existing constants/conventions
- Making tests too large or non-deterministic
- Skipping local validation and relying only on remote CI

## Reference Locations

- Pytest config: `pyproject.toml`
- Tests: `tests/`
- CI sources: `.buildkite/pipeline.yml`, `.buildkite/gpu_suites.py`
- Buildkite guide: `.buildkite/README.md`
- CI guide: `docs/en/developer_guide/ci.md`
