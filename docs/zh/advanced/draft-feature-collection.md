# 采集 external draft 所需的 target features

`--draft-feature-mode collect-only` 在已有 actor log-probability forward 中导出最终无 bias LM head 的选中输入，不训练或发布 draft 模型。默认 `off` 路径不会创建导出目录或 hook。

首版支持 TP=PP=CP=DP=1、无 VPP、micro-batch=1 和 actor 模型。关闭 `--keep-old-actor`、`--use-rollout-logprobs`，并设置 `--update-weights-interval 1`，避免 version counter 落后于 actor 更新。如果真实 batch 复用训练 forward，没有独立 actor log-probability forward，会在导出之前报错；采集不会补做 forward。同步 GRPO 可配置每轮多个 optimizer step，让正常 actor 路径本来就需要独立计算 log probabilities。

在已跑通的训练 recipe 中追加：

```bash
--draft-feature-mode collect-only \
--draft-feature-output-dir /local/experiments/draft-features \
--draft-feature-run-id run-001 \
--draft-feature-max-tokens 4096 \
--draft-feature-max-batches 8 \
--draft-feature-max-bytes 1073741824
```

每轮写入 `run-001/round-N/head.safetensors`、选中的 `batch-XXXX.safetensors` 和 JSON manifest。head 每轮只复制一次；features 不共享原激活 storage。先完成 tensor 文件，再 rename 公布 manifest。消费者忽略临时文件，只读完整 manifest。已有 round 目录会报错；重启使用新的 run ID。

预算按**每轮 tensor payload**计数：一次 head bytes，加上选中 token 数 × hidden size × 实际 feature 元素大小。JSON、序列化缓冲和文件系统开销不在预算内。多轮累计导出需要额外磁盘与明确的清理策略。整轮为空会带原因失败；已有 batch 后达到预算则成功并记录停止原因。`copy_seconds` 包括 CPU 传输、序列化和写文件，不等于纯 D2H 带宽测量。

token map 保留原始 sample、prompt group、rollout ID、source position 和 next-token target。padding 不产生选中行；排除 masked token 不修改训练 mask。通过 `vime.utils.draft_feature_contract.manifest_from_dict` 加载，显式还原嵌套 contract 并校验 offsets、shift、身份、dtype 与 bytes。

rollout provenance、target-forward version 和 head version 含义独立。缺失 rollout version 保留 unknown（`[]`），混合版本如实保存。actor/head label 使用本次运行的 weight updater counter。新 actor 先恢复模型和 optimizer，再做首次发布；counter 可以重新开始，不能当作全局 checkpoint 标识。发布错误会在 driver 下一轮之前传播。严格 on-policy 消费者仍需独立核实实际 rollout provenance。

## 可复现验证

CPU 回归不启动 Ray 或 CUDA：

```bash
python -m pytest -q tests/test_draft_feature_contract.py \
  tests/test_draft_feature_collector.py tests/test_draft_feature_metadata.py \
  tests/test_megatron_argument_validation.py
```

Ouro 验证入口 `tests/integration/run_draft_feature_ouro.py` 包装已有 `examples/ouro/train.py`，在该路径后传入正常 recipe 参数。它在 off/on 两组都使用同一 target forward，保留真实 RLT 生成、reward、优化与权重发布，并将参考 logits 写到消费产物目录之外。`--only-train-params-name-list lm_head.weight` 是较小显存的 smoke 配置，结果必须明确注明。

完整验证比较固定 replay 的 off/on、使用同版本导出 head 重建选中 logits、执行六轮真实生成/更新/保存/发布，再用新进程和新 run namespace 恢复。比较 loss、entropy、梯度、optimizer、RNG 与 checkpoint。用匹配重复实验报告采集开销；采集本身不承诺 rollout 加速。checkpoint、大 tensor 与原始日志放在源码目录外。
