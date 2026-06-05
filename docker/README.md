# Docker release rule

vime ships one image based on the official vllm image, published to
`inferactinc/public`:

- `vime-cu129-latest` — rolling pointer to the current image
- `vime-cu129-nightly-<date>` — immutable dated snapshot (e.g. `vime-cu129-nightly-20260603a`)

Build locally:

```bash
docker build -f docker/Dockerfile -t vime-cu129 .
```

## Release matrix

Before tagging a new stable image, the following matrix must pass. All four
are currently TODO — none has been wired into CI yet:

- [ ] Qwen3-4B sync
- [ ] Qwen3-4B async
- [ ] Qwen3-30B-A3B sync
- [ ] Qwen3-30B-A3B fp8 sync
