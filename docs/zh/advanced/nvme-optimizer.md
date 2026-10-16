# NVMe optimizer state 流式加载

NVMe streaming 用 optimizer step 时间换取 GPU 显存：FP32 main parameters 和 Adam
moments 保存在文件中，每次只加载、更新并写回一个 bucket。它与训练和 rollout 之间搬走
整个训练 actor 是不同的机制。本功能不增加整 actor 的 disk offload，也不支持 Muon。

## 启用 streaming

使用包含 `OptimizerConfig.defer_main_param_initialization` 的 Vime Megatron 补丁环境，
并为每个独立作业指定单独、可写的 NVMe 目录：

```bash
--optimizer adam \
--use-distributed-optimizer \
--stream-optimizer-state-to-disk \
--offload-train-disk-dir /root/nvme/job-NEW \
--offload-train-disk-chunk-mb 64 \
--stream-optimizer-state-moment-dtype fp32
```

不要让并发作业共享 scratch 目录：每个 rank 在初始化时会清理自己的 scratch 子目录。
延迟初始化保留 main tensor 的 shape、dtype、device 和对象身份，同时释放 CUDA storage。
最大的单个 FP32 shard 仍需短暂放入显存；随后逐个 materialize main-only bucket。
模型原生的 FP32 参数仍驻留 GPU。

支持范围为 BF16 模型训练 + Adam + Megatron DistributedOptimizer，不支持 precision-aware optimizer、
CPU optimizer offload、Megatron FSDP、FP8 模型参数、Muon 或 stateless Adam。
FP16 模型训练会提前报错，因为目前没有实现其 loss-scaler checkpoint 状态的保存与恢复。
FP32 moment storage 已有完整模型验证，BF16 moment storage 已有组件验证；
其他存储 dtype 未经过完整模型验证。

流式模式拒绝 `--reset-optimizer-states`（只重置原 optimizer，而不是流式状态）
和 `--load-main-params-from-ckpt`（会同时 materialize 全部 main bucket）。

## 保存与恢复

使用同步 `--ckpt-format torch_dist` 保存，并保持相同模型、optimizer state dtype 和并行拓扑；
其他 optimizer checkpoint 格式会被拒绝，包括自动识别出的旧格式 optimizer 状态。Checkpoint 包含各 rank
的 bucket 文件和 manifest；流式状态会先于 Megatron 发布 checkpoint tracker 保存。
改变 TP/PP/DP/CP 布局时不会自动 reshard。

不含流式 optimizer state 的 checkpoint 不能静默恢复该状态。`--no-load-optim` 表示明确
接受重新初始化 optimizer，而不是 checkpoint roundtrip。与 Megatron 一致，`--finetune`
和 release checkpoint 也跳过 optimizer 状态恢复；普通的 iteration-zero checkpoint
仍会恢复 optimizer 状态。上述只加载模型的模式仍可读取旧格式的模型权重。增加训练的 rollout 总量时，
若需要沿用保存的 scheduler，可使用 Megatron 的 `--use-checkpoint-opt-param-scheduler`。

## 可变 global batch

加入 `--variable-global-batch-size` 可保留尾部不足一个完整 batch 的 optimizer step。
例如 24 个样本配合 `--global-batch-size 16` 会执行 16 和 8 两步。
显式指定 `--global-batch-size-schedule 16,8` 也会启用可变模式。
不使用这些参数时，原有固定 batch 行为不变。

Batch 大小按 rollout group 计数，而不是 compact/subagent rollout 产生的训练样本数。
显式 schedule 必须恰好覆盖所有 rollout group。每一步（包括尾部不足完整 batch 的一步）
仍需提供足够的样本和 microbatch，以满足 Vime 的 DP/VPP 对齐约束；过小的 step 会报错，
而不是静默丢弃或补齐样本。

## 验证

在已准备好模型和数据缓存的独立八卡 Vime 容器中运行：

```bash
export NCCL_NVLS_ENABLE=0
export TMPDIR=/root/nvme
python -m pytest -q tests/fast-gpu/test_nvme_optimizer_main_init.py
python -m pytest -q tests/fast-gpu/test_nvme_stream.py
torchrun --standalone --nproc_per_node=8 -m pytest -q \
  tests/fast-gpu/test_nvme_stream.py -k bucket_fetch_step_matches
python tests/test_qwen3_4B_ckpt.py \
  --save-optimizer nvme --load-optimizer nvme \
  --skip-prepare --checkpoint-dir /root/nvme/checkpoint-NEW
```

`tests/test_qwen3_4B_nvme_benchmark.py` 提供手动 capture、replay、compare 和短期训练阶段，
使用 `--help` 查看参数。其组内合成奖励用于确保梯度非零，不是任务效果评测。
使用 Megatron stub 的 CPU 测试和真实 Megatron/GPU 测试应在不同 Python 进程中运行。

这些启动器会管理 Ray/vLLM 进程，应使用独立容器。GPU 空闲检查不等于资源预约。
耗时与显存取决于模型、拓扑、文件系统和页缓存；这些测试不证明裸 NVMe 带宽、
长期耐久性或断电恢复原子性。
