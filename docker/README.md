# Docker release rule

vime ships one image based on the official vllm image, published to
`inferactinc/public`:

- `vime-latest` — rolling pointer to the current image
- `vime-nightly-<date>` — immutable dated snapshot (e.g. `vime-nightly-20260603a`)

Build locally:

```bash
docker build -f docker/Dockerfile -t vime .
```

## Release matrix

Before tagging a new stable image, the following matrix must pass. All four
are currently TODO — none has been wired into CI yet:

- [ ] Qwen3-4B sync
- [ ] Qwen3-4B async
- [ ] Qwen3-30B-A3B sync
- [ ] Qwen3-30B-A3B fp8 sync
