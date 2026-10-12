# Docker release rule

vime ships one image based on the official vllm image, published as
`vllm/vime:latest`. Supports GB200/300 and H100/200.

Build locally:


```bash
just release
```

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-106B-A12B sync

## ROCm

`docker/Dockerfile.rocm` builds the AMD image, published as `vllm/vime-rocm:latest`
and consumed by `.buildkite/pipeline-rocm.yaml`. It layers on
`vllm/vllm-openai-rocm:nightly` — AMD's vLLM ROCm nightly, which already carries
ROCm 7.2.x, a ROCm PyTorch build, triton, aiter and flash-attn — and adds only the
training half: TransformerEngine (built from `ROCm/TransformerEngine`), apex,
Megatron-LM plus the `docker/amd_patch/` patches, torch_memory_saver, and vime.

Targets gfx950 (MI350X/MI355X); pass `--build-arg GPU_ARCH=` for another CDNA arch.

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vime-rocm:latest .
```

The vLLM patches under `docker/patch/` are cut against the CUDA base, which sits
at a different point on vLLM main than the ROCm nightly — ahead of it on some
files, behind on others. The ROCm build therefore applies them with
`patch --fuzz` and fails if any hunk was rejected. When that check trips, rebase
`docker/patch/latest/` rather than loosening the check.

Two things the CUDA image gets that this one does not, because the base does not
carry the vLLM code they are written against:

- the delta sampling-mask hunk in `vllm.patch` (needs `SamplingMaskLists.merge`),
  so ROCm keeps emitting sampling masks only on finish;
- `vllm-pd-request-metrics.patch`, which targets a protocol module this base
  splits differently, so per-request PD telemetry is absent. vime reads those
  fields defensively, so rollouts are unaffected.

Both are guarded: the build fails once the base converges, at which point the
skip is removed rather than kept.
