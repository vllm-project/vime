# Docker

vime ships one image, based on `vllm/vllm-openai:v0.21.0-cu129-ubuntu2404`:
- Published: `aosheninferact/vime-vllm:cu129` (also `latest`)
- Megatron pin: `1dcf0dafa884ad52ffb243625717a3471643e087` + `docker/patch/latest/megatron.patch`

Build locally:

```bash
docker build -f docker/Dockerfile -t vime/pr-9-vllm:cu129 .
```
