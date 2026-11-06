---
name: release
description: Prepare and verify a vime release, including version bumps, pinned Docker dependencies, patch coverage, and release-specific validation. Use when cutting or auditing a vime release.
---

# Release vime

Prepare a release without publishing, tagging, pushing images, or changing the
dependency baseline unless the user explicitly requests those external or
scope-expanding actions.

## Establish the release baseline

- Inspect the worktree and preserve unrelated user changes.
- Compare the previous release tag and release commit to identify the current
  repository conventions.
- Confirm the requested vime version, the immutable `BASE_IMAGE` digest and
  the `VLLM_COMMIT` pin in `docker/Dockerfile`.
- Do not pull in an unmerged vLLM/Docker upgrade merely because a newer
  branch exists. Treat that as a separate decision.

## Update release versions

- Set the package version in `setup.py`.
- Set the documentation version in `docs/conf.py`.
- Bump `docker/version.txt` to a unique image version following its existing
  dated naming convention.
- Search the repository for the old vime version and review every remaining
  occurrence instead of replacing unrelated dependency versions.

## Audit the Docker patch stack

- Read `PATCH_VERSION` from the Dockerfile; its selected directory is the
  release patch stack. Vime does not require a separate Conda/stable snapshot.
- Check each Docker stage's copied and applied patches, including the Rust
  frontend, router and conditional ARM SDK stage. All selected patches must
  have an installation path; copying a patch does not prove it is applied.
- Verify patches apply in Dockerfile order to clean checkouts of the pinned
  upstream commits. Do not validate against a dirty developer checkout.

## Audit Docker dependencies

Use `docker/Dockerfile` and `docker/justfile` as the build sources. Check at least:

- vLLM base digest, commit and CUDA variant;
- Megatron, torch-memory-saver, FlashQLA, and other shared source pins;
- consistent source pins across Docker stages, `PATCH_VERSION`, patch filenames,
  application order, optional patches and failure-on-conflict behavior;
- PyTorch, torchvision, torchaudio, CUDA Python, Transformer Engine, router,
  NumPy, and SciPy pins;
- whether dependency resolution can undo a compatibility pin later in the
  script; reassert and validate such pins after the resolving install;
- architecture-specific dependency builds and conditional patches.

Prefer direct loops over duplicated patch-application blocks while preserving
required-versus-optional semantics and useful failure messages.

## Validate before handoff

Run the checks that are available locally:

- `python .claude/skills/release/scripts/check_release.py --repo .
  --expected-version <version>`;
- `python setup.py --version`;
- patch parsing plus clean-checkout application against the pinned upstream
  commits;
- `git diff --check` and a final review of the complete release diff.

The metadata checker is a static preflight, not proof of clean patch application,
successful Docker builds or GPU correctness. Run relevant CPU/GPU CI and native
AMD64/ARM64 Docker verification for the requested release scope. Explicitly report
any build or GPU validation that was not run. Vime has no Conda release workflow.
