# 同一作业内的 TransferQueue 轨迹传输

可选 `simple-storage` 模式将完整的 text prompt group 持续写入 CPU SimpleStorage，其他生成请求可继续运行。复用原有 GRPO 生成、reward/filter 与 batch conversion。优化仍在完整固定轮次封存后开始；首版不与下一策略的生成重叠。

默认 `--transfer-queue-mode off` 不导入 TransferQueue，也不创建其 actor、client 或 heartbeat。开启时要求运行环境已有可选依赖 `TransferQueue==0.1.9`（`transferqueue` package extra）。支持单 text actor、TP=PP=CP=DP=1、无 VPP、GRPO、标准同步 `train.py` 和可保存状态的 global dataset。拒绝 critic、partial rollout、fault recovery、debug replay、自定义 conversion/轮后变换、multimodal、MTP 与 MoE replay。codec 会在写入之前拒绝不支持的 Sample 字段。

在已跑通的同步 recipe 中追加：

```bash
--transfer-queue-mode simple-storage \
--transfer-queue-job-id training-001 \
--transfer-queue-restart-epoch 0 \
--transfer-queue-max-groups 64 \
--transfer-queue-max-tokens 131072 \
--transfer-queue-max-bytes 268435456 \
--transfer-queue-timeout-s 300 \
--transfer-queue-lease-s 120
```

group 容量必须装下整轮 rollout batch。写 tensor 前预留 token/bytes；超大 group 或轮次报错，不无限等待。bytes 计入五份 tensor/传输缓冲、owned JSON 与 decoded Sample 字段。一次只执行一个 codec/write/read；coordinator 只保留 descriptor 和 checksum，不再复制 payload。封存校验逐个处理 group。Ray/allocator metadata 和模型激活不在 working-set 预算内；同时记录进程 RSS 与 `peak_working_bytes`。没有 GPU zero-copy，也不保证进程 RSS 硬上限。

## 身份、版本与消费

每个完整 group/attempt 使用本 job、restart epoch 下的独立 partition。初次 put 不向共享 partition 追加。encode→storage→decode 保留原 group/sample/rollout 身份、reward、mask、log probabilities 与 ragged top-p。Sample 的 nullable rollout ID 保持 nullable；contract 采用 VIME 现有 sample-index fallback，不用 consumer 轮次补造身份。

真实 payload 读取成功并通过 checksum 后才可 READY。完全相同的重复发布幂等，冲突与迟到 group 根据已提交轮次 watermark 拒绝。所有 child 都必须带与 serving cohort 确认版本相同的实际 policy provenance；缺失或混合版本直接拒绝，不从 driver 补齐。

封存后停止 admission、暂停 serving engine，再由 maintenance drain probe 确认 idle。UNKNOWN/BUSY 或部分发布失败会阻止该 opt-in 路径继续接收。下一轮要求所有 serving engine 确认同一个已变化的版本。

coordinator 执行 READY→LEASED→PREPARED→TRAINING→TRAINED→COMMITTED，独立 heartbeat 在完整 actor 训练计划期间续租。读取 metadata、conversion 或一个 microbatch 都不是训练确认。LEASED/PREPARED 过期后可用新 generation 接管；TRAINING/TRAINED 过期或状态未知则进入 UNKNOWN 并停止。不会自动重放 optimizer，也不承诺跨崩溃 exactly-once。

提交后先删除 payload，再释放容量。删除失败保留待 GC 状态并停止推进，不撤销 optimizer，也不重训。`reclaim_committed()` 仅重试本 job 的删除，不恢复失败训练计划。读写失败暂停 admission 并尝试清理本 partition；超时远端写入可能迟到完成，其旧 epoch 永不作为新训练输入复用。

## 保存、恢复与验证

保存等待同步 native checkpoint、data-source 保存、发布确认与安静队列。`transfer-queue.json` 记录共同 next rollout cursor、sampling digest、job 与提交 watermark。使用同一 job ID、严格更大的 restart epoch 从共同 checkpoint 重启。游标或配置不匹配时，在创建 queue actor 前报错。SimpleStorage 是内存传输，不是 durable optimizer 事务；不完整的 model/data/queue snapshot 不能作为共同 checkpoint 恢复。

```bash
python -m pytest -q tests/rollout/test_transfer_queue_contract.py \
  tests/rollout/test_transfer_queue_adapter.py tests/test_megatron_argument_validation.py
python -m pytest -q tests/integration/test_transfer_queue_backend.py
```

backend 测试使用真实 controller/storage/client actor，覆盖 streaming、lease/commit、超时清理、GC 重试和新 epoch 恢复。其中 policy/训练完成事件是 fixture，不是 GPU 模型 E2E。完整 E2E 必须经过 vLLM→TQ→Megatron→optimizer→下一 serving version，再做新进程恢复并与 queue-off 对照。匹配重复实验报告 `put_s`、`get_s`、payload bytes、peak working bytes、RSS 和总轮次时间。生成/优化屏障可能增加开销；不承诺加速。

只关闭本 job 的 client，让所属进程正常退出。不调用全局 TransferQueue close，不停止共享 Ray 服务，不删除其他 job 的 partition。
