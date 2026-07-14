# 昇腾 W8A8-MXFP8 Rollout

Vime 通过 vLLM 和 vLLM-Ascend 支持昇腾 950 上的 W8A8-MXFP8 rollout 在线权重更新。
训练权重通过 IPC 或 NCCL 传输时保持 BF16；每个 rollout worker 在 vLLM 加载权重前，
在线量化符合条件的 Linear 和 FusedMoE 权重，随后由 vLLM-Ascend 恢复推理布局。

## 支持矩阵

| 能力 | 状态 |
| --- | --- |
| 昇腾 950 W8A8-MXFP8 rollout | 支持 |
| Colocated IPC 在线更新 | 支持 |
| Decoupled NCCL 在线更新 | 支持 |
| ModelSlim 描述的 Linear 和 FusedMoE 权重 | 支持 |
| MXFP4 | 本特性暂不支持 |
| 外部独立 vLLM 服务 | 本特性暂不支持 |
| 磁盘/delta 权重重载 | 本特性暂不支持 |

## 版本基线

本适配以本地 `vllm-ascend/main` 提交 `8bedc666` 的 API 为基线，依赖：

- `AscendModelSlimConfig`；
- Linear 和 FusedMoE 的 `W8A8_MXFP8` scheme；
- `torch_npu.npu_dynamic_mx_quant`；
- vLLM checkpoint-format 权重更新事务和 worker extension API。

昇腾 950 环境需要安装相互兼容的 vLLM、vLLM-Ascend、torch-npu、CANN 和 ModelSlim。
Vime 仓库本身不安装昇腾软件栈。

## 模型准备

Rollout checkpoint 目录必须包含 ModelSlim 生成的 `quant_model_description.json`。
请使用与模型及 vLLM-Ascend 环境匹配的 ModelSlim 版本生成完整文件。目标量化层必须标记为
`W8A8_MXFP8`，group size 必须为 32。完整的模型专用文件会包含类似以下的元数据和条目：

```json
{
  "quant_method": "ascend",
  "group_size": 32,
  "model.layers.0.self_attn.q_proj.weight": "W8A8_MXFP8",
  "model.layers.0.self_attn.k_proj.weight": "W8A8_MXFP8",
  "model.layers.0.input_layernorm.weight": "FLOAT"
}
```

以上片段仅用于说明，并不是任一模型的完整描述。必须按照当前 vLLM-Ascend ModelSlim
解析器要求的名称覆盖所有必要层。缺失或不兼容的描述文件会导致 vLLM-Ascend 拒绝
`--quantization ascend`。

初始 rollout checkpoint 可以包含 MXFP8 推理权重，但 Vime 在线发送的 actor 更新仍为 BF16；
不要在 trainer 侧对这些更新权重做 MXFP8 量化。

## 配置方法

使用现有的 vLLM 量化参数启用该特性：

```bash
VLLM_ARGS=(
  --vllm-quantization ascend
  --vllm-gpu-memory-utilization 0.7
)
```

### Colocated IPC

在 Vime 启动命令中加入 `--colocate`：

```bash
python train.py \
  --colocate \
  --hf-checkpoint /path/to/rollout-checkpoint \
  --vllm-quantization ascend \
  --vllm-gpu-memory-utilization 0.7 \
  ...
```

Vime 会安装 MXFP8 worker extension，并选择 vLLM 原生 IPC 权重传输。BF16 命名张量通过
IPC 共享，在各 rollout worker 内完成在线量化。

### Decoupled NCCL

使用普通的非 colocated 拓扑，不传 `--colocate`：

```bash
python train.py \
  --hf-checkpoint /path/to/rollout-checkpoint \
  --vllm-quantization ascend \
  --vllm-gpu-memory-utilization 0.7 \
  ...
```

Vime 会安装同一个 MXFP8 worker extension，并选择 vLLM 原生 NCCL 权重传输。两种模式仅
传输方式不同；在线量化和布局处理都在 worker 内执行，逻辑完全复用。

## 权重更新生命周期

两种传输模式都执行以下事务：

1. 启动 vLLM checkpoint-format 权重更新，恢复模型格式元数据并保留运行时张量存储。
2. 准备 Vime MXFP8 worker extension，包装统一的 `model.load_weights()` 入口。
3. 通过 IPC 或 NCCL 传输 BF16 张量。
4. 使用 `npu_dynamic_mx_quant` 在线量化目标权重，同时加载 FP8 权重和对应的
   `*_scale` 张量。
5. 完成 vLLM 原生 layerwise processing，并恢复 vLLM-Ascend MXFP8 推理布局。
6. 仅在 finish 和 finalization 都成功后发布新的 Vime weight version。

其他量化配置保持现有 vLLM 行为。

## 昇腾 950 验收步骤

以下步骤必须分别在目标昇腾 950 环境的 IPC 和 NCCL 模式执行：

1. 使用 `--update-weights-interval 1` 和上述 MXFP8 参数启动一个小型 Vime 任务。
2. 确认 vLLM-Ascend 成功加载 ModelSlim 描述并进入健康状态。
3. 至少完成两轮 actor 更新和 rollout。
4. 确认每轮 start/update/finish 事务中没有缺失权重、缺失 scale、shape 或 dtype 错误。
5. 确认每次更新后 rollout 生成成功，且 Vime 报告的 weight version 持续递增。
6. 使用 `--colocate` 完成一次 IPC 验收，再去掉该参数完成一次 NCCL 验收。

本仓库的 CPU 测试通过 stub 验证配置、量化转换、生命周期和传输契约，不能替代上述硬件测试。

## 故障排查

`ModelSlim Quantization Config Not Found`
: 确认 `--hf-checkpoint` 指向的目录包含 `quant_model_description.json`，且该文件针对当前模型生成。

缺失 `weight_scale` 或 shape 不匹配
: 检查 ModelSlim 文件中的所有量化层名和融合模块映射；确认 `group_size` 为 32，且安装的
  vLLM-Ascend 与本文基线 API 兼容。

Worker extension 冲突
: 移除自定义 `--vllm-worker-extension-cls`。本特性必须使用 Vime 的 MXFP8 extension；Vime
  会直接报错，不会静默替换已有扩展。

外部服务被拒绝
: 让 Vime 自行拉起 rollout server。本特性不会向独立启动的外部 vLLM 服务注入 MXFP8
  worker extension。
