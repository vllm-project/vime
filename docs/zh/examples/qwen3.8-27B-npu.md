# 在 Ascend 上运行 Qwen3.8-27B

使用 [NPU 镜像及依赖补丁](../../en/get_started/NPU.md)。该 checkpoint 沿用
Qwen3.5 稠密混合注意力架构，复用已有模型桥。将 `Qwen/Qwen3.8-27B` 下载到
`/root/models/Qwen3.8-27B`，将 `zhuzilin/dapo-math-17k` 下载到
`/root/datasets/dapo-math-17k`，然后运行：

```bash
bash scripts/run-qwen3.8-27B-npu.sh
```

默认使用 16 个逻辑 NPU：8 个进行 TP4/PP2 训练，8 个运行两个 TP4 推理引擎。
运行 100 轮 GRPO，每轮 32 题 × 8 个样本，回答上限 4096，V2 runner、非 MTP、
CPU optimizer offload、full/uniform/1 重计算。可覆盖 `MODEL_DIR`、`DATASET_DIR`、
`CHECKPOINT_DIR` 和 `NUM_ROLLOUT`。每 25 轮保存模型及优化器状态，四份 checkpoint
合计约需 1.4 TiB，模型和数据集空间另计。

`--qwen-gdn-tensor-parallel` 按 head 切分 GDN 的 K/V、投影、卷积及逐 head 状态。
输入及输出归一化参数仍共享，其梯度在 TP 组内求和。HF 加载、权重回传和分布式
checkpoint 均分别处理不等长的 Q/K/V 分段。支持开启或关闭 SP，目前要求 CP=1。
该开关默认关闭，不改变原有复制计算路径。

启动器建立独立 Ray 会话，不停止其他作业。在共享主机上选择空闲的
`RAY_PORT_BASE`（默认 18479，使用到 base+1099）及 HCCL 端口段，并使用独立
NPU 容器。脚本不清理已有 Ray 集群，也不自动恢复旧 checkpoint。
任务结束后，该脚本建立的 Ray head 保留，供检查日志和状态。

端到端 smoke 用例将输入下载到 `HF_HOME`，使用 16 个 NPU 运行两轮短 GRPO，
覆盖原生 HF 加载、head TP/SP/PP、训练、HCCL 权重更新及后续生成，不启用 MTP：

```bash
python tests/test_qwen3.8_27B_npu.py
```

用例注册在 Buildkite NPU `smk` suite，需要在 NPU gate 选择 `smk`。
CPU 或 pre-commit 通过不代表 NPU 用例已经执行。
