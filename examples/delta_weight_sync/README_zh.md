# Ascend 增量权重同步

此示例用于非 colocate 场景。训练端只把发生变化的 safetensors 字节发布到共享文件系统；
每个 rollout host 将 delta 原地应用到本地 HF checkpoint，再由 vLLM 热加载。

```bash
--update-weight-mode delta \
--update-weight-transport disk \
--update-weight-disk-dir /shared/vime-delta \
--update-weight-local-checkpoint-dir /local-nvme/vime-rollout-checkpoint \
--update-weight-delta-encoding xor \
--update-weight-delta-checksum xxh3-128
```

`--update-weight-disk-dir` 必须是训练端和所有 rollout host 可见的 Linux POSIX
共享文件系统；本地 checkpoint 路径应为 host-local 存储，并在首次同步时从
`--hf-checkpoint` 初始化。对象存储挂载需要显式 publish/refresh 时，可分别使用
`--custom-update-weight-post-write-path` 与 `--custom-update-weight-pre-read-path`。
本地磁盘至少需要容纳一份完整的 HF safetensors checkpoint 及首次复制时的临时空间；训练端和
所有 rollout 镜像还必须安装 `blake3`、`xxhash`、`zstandard`，并应用随本仓库提供的
vLLM/vLLM-Ascend patch。
