# Docker release rule

vime ships a single image variant, based on the official vllm release
(`vllm/vllm-openai:v0.21.0-cu129-ubuntu2404`), bundling the slime training
stack (Megatron-LM, mbridge / Megatron-Bridge, modelopt, apex, flash-attn 2 +
flash-attn 3 hopper, TE 2.10, flash-linear-attention, tilelang,
torch_memory_saver).

current stable version:
- vllm 0.21.0 (cu129) + megatron `1dcf0dafa884ad52ffb243625717a3471643e087` + slime patch `docker/patch/latest/megatron.patch`

Published image:
- `aosheninferact/vime-vllm:cu129` (also tagged `latest`)

Build locally:

```bash
docker build -f docker/Dockerfile -t vime/pr-9-vllm:cu129 .
```

History (sglang-based images, predates the vllm switch):
- sglang v0.5.9 (bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862
- sglang v0.5.7 nightly-dev-20260107-dce8b060 (dce8b0606c06d3a191a24c7b8cbe8e238ab316c9), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862
- sglang v0.5.6 nightly-dev-20251208-5e2cda61 (5e2cda6158e670e64b926a9985d65826c537ac82), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)
- sglang v0.5.5.post1 (303cc957e62384044dfa8e52d7d8af8abe12f0ac), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)
- sglang v0.5.0rc0-cu126 (8ecf6b9d2480c3f600826c7d8fef6a16ed603c3f), megatron 48406695c4efcf1026a7ed70bb390793918dd97b
