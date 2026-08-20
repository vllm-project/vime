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
per test. GPU tests use `vllm/vime:latest`; rebuild and publish that image
before validating a Dockerfile or vLLM patch change.

## Registering tests

- Add always-on CPU tests to the appropriate command in
  `.buildkite/pipeline.yml`.
- Add GPU tests to a suite in `.buildkite/gpu_suites.py` and update the suite
  count shown by `.buildkite/pipeline.yml`.
- Keep `.buildkite/README.md` synchronized with pipeline behavior.

Run the exact command locally before triggering its remote Buildkite job. For
GPU failures, reproduce on an H200 node with the same image and environment,
then rerun the remote suite only after the local test passes.
