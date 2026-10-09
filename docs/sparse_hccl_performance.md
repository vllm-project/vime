# 稀疏 HCCL：代码逻辑、正确性与模型适配说明

本次提交已移除分阶段计时功能和耗时统计。保留 EP 正确性修正、按实际载荷长度传输、空载荷跳过及缓存复用；checksum 与正确性校验不变。

## 接口流程与主要改动

`Bridge 导出/差分 → 训练端 gather/打包 → checksum → HCCL → 接收端 checksum → PP/EP/TP 归属处理 → 索引写入或原生加载回退 → 本地校验 → 恢复生成`

- 发送端：`update_weight_from_sparse_hccl.py::update_weights` 首次发送 dense seed 并建立快照，之后 `_send_sparse_delta` 导出变化索引和数值；`_publish_flush` 提交接收端 RPC，并保持缓冲区存活到传输和写入完成。
- 聚合：`delta_sync/sparse_gather.py::gather_slot_entries_to_rank0` 保留数量交换和合并顺序，HCCL P2P 使用实际长度并跳过空载荷；其他后端保留填充路径。bucket/workspace 仅在缓存未命中时创建。
- 接收端：`sparse_hccl_engine.py::receive_weights` 检查 checksum 并过滤非本 PP 权重，调用 `apply_sparse_hf_patches_with_loader`。
- EP：`sparse_weight_patch.py::_apply_direct_moe_patches` 检查全局专家编号，映射到本地槽位后再检查参数存储范围。动态 EPLB 在静态归属过滤之前回退，避免丢弃权重。
- 未知或不满足直接写入条件的布局使用 NaN 掩码加模型原生 `load_weights`，不修改传输框架和 patch 格式。

## 其他模型的泛化能力

这里的“泛化”指权重同步实现能否迁移到其他模型，不是模型推理或任务效果的泛化。本节依据当前代码判断适配边界，没有新增其他模型的实测结果。

### 通用部分与模型专用部分

- **通用传输层：** 稀疏索引/数值打包、checksum、HCCL 实际长度传输、空载荷跳过和缓存复用不依赖具体模型名称，可供其他模型复用。但传输层可复用不等于端到端已经支持。
- **导出适配：** `HfWeightIteratorSparseBridge.__init__` 使用 `AutoBridge.from_hf_pretrained`，`get_hf_weight_chunks` 使用 `export_hf_weights`。迁移前必须确认当前 Bridge 版本支持该模型，并正确导出名称、形状、融合拆分和专家编号；不能仅依据 HF 模型可加载就认定稀疏更新可用。
- **原生加载回退：** 实际接收入口 `apply_sparse_hf_patches_with_loader` 先校验 patch，再尝试 MoE 直接写入；剩余 patch 在 NaN 掩码上下文中调用 `model.load_weights`。这提供了复用模型原生 TP/PP 加载逻辑的途径，但仍须验证该模型加载器的复制、转换、融合和专家分片行为与掩码兼容。回退可能重新构造较大的稠密临时张量，网络稀疏不代表接收端计算和内存也完全稀疏。
- **Qwen3 专用优化：** `_apply_direct_moe_patches` 明确要求 `model.config.model_type == "qwen3_moe"`，并匹配 `experts.<编号>.(gate_proj|up_proj|down_proj).weight`。因此本次 EP 直接写入收益不能直接推广到其他 MoE。`partition_qwen3_sparse_patches`、`_qwen3_tp_contribution_mask` 和 `_map_qwen3_stacked_name` 也含 Qwen3 命名/布局假设，不能当作任意模型的通用映射。部分辅助映射并非实际 HCCL 接收入口，不应将其存在视为端到端支持证据。

代码证据位置：发送端 `vime/backends/megatron_utils/update_weight/hf_weight_iterator_sparse_bridge.py`；接收端 `vllm_ascend/distributed/weight_transfer/sparse_weight_patch.py`（随 `docker/npu_patch/vllm-ascend.patch` 提供）。

| 模型或配置 | 当前可复用能力 | 尚需确认的部分 | 证据等级 |
| --- | --- | --- | --- |
| Qwen3-30B-A3B，本报告配置 | 通用传输、静态 EP 直接写入、原生加载 | 整模型全量参考一致性仍未证明 | 已有端到端与局部正确性实测 |
| Qwen3-235B 等同架构规模变体 | 条件满足时可复用 Qwen3 MoE 路径 | Bridge、专家/注意力配置、内存容量、双机拓扑、单参数索引范围和性能 | 架构层面候选，未实测 |
| Qwen3 dense、Llama 类 dense 模型 | 通用传输和原生加载回退是候选路径 | 导出映射、融合 QKV/MLP、TP/PP、共享 embedding/输出头及加载器掩码行为 | 代码层面候选，未实测 |
| Qwen2 MoE、Mixtral 等其他 MoE | 通用传输可复用 | 模型专用专家名称、全局/本地编号、融合布局、expert TP；不进入本次 Qwen3 MoE 快路径 | 需要适配与验证，未实测 |
| DeepSeek 类模型 | 通用传输可复用 | MLA、共享/路由专家、特殊融合参数及量化布局；不能直接套用 Qwen3 映射 | 需要专项评估，未实测 |
| 量化、NZ 特殊布局、动态 EPLB | 部分路径具有安全检查或原生回退 | 量化尺度与打包权重、逻辑/物理坐标、动态专家归属和完整加载行为 | 回退不等于支持，未完成端到端认证 |

表注：“可复用”是当前代码的机制分析，不是已通过测试的模型支持清单。“未实测”表示本次没有对应模型的运行结果，也没有可引用的性能提升百分比；模型名称仅表示待验证类别。Qwen3-235B 同架构也不能继承 30B 的耗时、显存或双机结论。

### 泛化的硬约束与性能边界

接收端使用 int32 稀疏索引，`_validate_sparse_hf_patches` 和 `_validate_sparse_hf_patch` 明确拒绝单个 HF 参数元素数达到或超过 `2**31` 的情况。限制针对单参数，不是模型总参数量；大模型需逐参数检查，不能仅凭“235B”判断通过或失败。packed/量化和非标准 NPU 布局不能默认套用逻辑扁平索引。

性能收益取决于真实变化比例，而不是模型是否叫 MoE：稀疏值还附带索引；变化密集时，索引和聚合成本可能抵消网络收益。若其他模型大量进入稠密化原生回退，接收端开销也可能高于本报告配置。不能把 Qwen3 的优化效果直接推广到其他模型。

建议新增模型按以下顺序认证：小张量映射测试（含未修改值）→ TP/PP 分片及 MoE EP 专家归属测试 → 同一初始权重、同一更新下与 Full 逐参数比较 → 端到端运行与资源检查；目标为双机时另做双机通信与一致性验证。本次仅补充评估说明，未执行这些新增模型实验。


## 正确性与验证边界

移除计时后重新执行的 218 项回归全部通过，包含真实 NPU 专家映射与 TP 场景、修改和未修改测试张量、动态 EPLB 回退，以及旧计时参数被拒绝的检查。此前独立四卡 HCCL 的 20 组测试覆盖空载荷、不等长和暂存区复用；本次未修改该聚合实现，未重跑这组四卡测试。

已有 Qwen3-30B-A3B 单机 TP2/PP2/EP2 端到端验证，不代表双机或 Qwen3-235B 已认证。测试张量全量比较及运行时本地修改条目校验，不等于整个 30B 模型与 Full 参考结果逐参数一致。
