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

`docker/Dockerfile.rocm` builds the AMD image, published as `vllm/vime-rocm:latest`
and consumed by `.buildkite/pipeline-rocm.yaml`. It layers on AMD's vLLM ROCm
nightly image for the vLLM commit that `docker/Dockerfile` pins
(`vllm/vllm-openai-rocm:nightly-rocm100-<VLLM_COMMIT>`, by digest), which already
carries ROCm 10 (`rocm_sdk` wheels), a ROCm PyTorch build, triton, aiter and
flash-attn, and adds only the training half: TransformerEngine (built from
`ROCm/TransformerEngine`), apex, Megatron-LM plus the `docker/amd_patch/` patches,
torch_memory_saver, and vime.

Targets gfx950 (MI350X/MI355X); pass `--build-arg GPU_ARCH=` for another CDNA arch.

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vime-rocm:latest .
```

Because both images sit on the same vLLM commit, the ROCm build applies the same
`docker/patch/latest/` vLLM patches as `docker/Dockerfile`, in the same order and
with the same strict `git apply`. Move `VLLM_COMMIT` and the ROCm `BASE_IMAGE`
together; `tests/test_docs_consistency.py` fails when they diverge.
