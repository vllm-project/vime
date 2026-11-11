# Docker release rule

vime ships one image based on the official vllm image, published as
`vllm/vime:latest`. Supports GB200/300 and H100/200.

Build and publish from the `docker/` directory:


```bash
just build
```

Run `just build` on both native AMD64 and ARM64 hosts. After both digests are
available, publish the version tag and `latest`:

```bash
just manifest "" sha256:AMD64_DIGEST sha256:ARM64_DIGEST
```

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-106B-A12B sync

## ROCm

`docker/Dockerfile.rocm` builds `vllm/vime-rocm:latest` for gfx950
(MI350X/MI355X), using a digest-pinned vLLM ROCm base image.
It is used by `.buildkite/pipeline-rocm.yaml`.

Build from the repository root:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vime-rocm:latest .
```

Override `GPU_ARCH` with `--build-arg` for another supported architecture.

The build applies shared vLLM patches from `docker/patch/latest/` with strict
`git apply`, in CUDA's order; ROCm-only patches live in `docker/amd_patch/`.
Megatron-LM uses CUDA's `MEGATRON_COMMIT` and the shared `megatron.patch`.
Keep `BASE_IMAGE` compatible with the shared patches.
