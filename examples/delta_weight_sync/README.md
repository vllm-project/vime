# Ascend Delta Weight Sync

This example enables non-colocated disk delta weight sync for Ascend rollout engines.
The trainer writes only changed safetensor bytes to a filesystem shared with rollout
hosts; each host applies them to a local HF safetensors checkpoint and vLLM reloads it.

```bash
--update-weight-mode delta \
--update-weight-transport disk \
--update-weight-disk-dir /shared/vime-delta \
--update-weight-local-checkpoint-dir /local-nvme/vime-rollout-checkpoint \
--update-weight-delta-encoding xor \
--update-weight-delta-checksum xxh3-128
```

Only non-colocated rollout is supported. `--update-weight-disk-dir` must be a Linux
POSIX filesystem visible to training and every rollout host. The local checkpoint
directory is host-local storage and is seeded from `--hf-checkpoint` on the first sync.
Reserve enough local capacity for one complete HF safetensors checkpoint plus temporary
copy space during the initial seed. Both the trainer and every rollout image must include
`blake3`, `xxhash`, and `zstandard`, and use the accompanying vLLM/vLLM-Ascend patches.
Use `--custom-update-weight-post-write-path` and
`--custom-update-weight-pre-read-path` for object-storage mounts that need explicit
publish/refresh operations.
