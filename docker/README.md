# Docker release rule

vime ships one image based on the official vllm image, published as
`aosheninferact/vime-vllm:cu129` (also `latest`).

Build locally:

```bash
docker build -f docker/Dockerfile -t vime/pr-9-vllm:cu129 .
```

## Release matrix

Before tagging a new stable image, the following matrix must pass. All four
are currently TODO — none has been wired into CI yet:

- [ ] Qwen3-4B sync
- [ ] Qwen3-4B async
- [ ] Qwen3-30B-A3B sync
- [ ] Qwen3-30B-A3B fp8 sync
