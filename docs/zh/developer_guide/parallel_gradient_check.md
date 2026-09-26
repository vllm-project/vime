# 并行回放的逐参数梯度诊断

全局梯度范数无法发现符号翻转或元素置换。可选参数
`--ci-save-parameter-grads DIRECTORY` 在反向梯度归约完成后、优化器准备、
unscale 和裁剪之前保存所有可训练 dense 参数的梯度，同时保留原有范数检查。

各 rank 保存到 `DIRECTORY/ROLE/rollout-N/step-M/rank-R.pt`。
每次运行必须使用新的共享目录。保存操作不增加 collective，不修改梯度缓冲区，
但会增加同步的 GPU→CPU 拷贝和磁盘写入，适合小模型正确性测试。
离线比较器需要在 CPU 内存中载入快照及两次运行重建后的张量。

distributed optimizer 的 reduce-scatter 完成后，每个 DP rank 只有一部分 bucket
包含有效归约结果。诊断只保存该范围与参数范围的交集，而非整个 `main_grad`。
比较器检查分片完整性，重建 TP 行列切分与 gated MLP 排列，使用 PP 全局层号，
并检查重叠副本。缺少 rank、参数或 step，以及非有限值、shape 不匹配都会失败。
数值错误显示参数名、逻辑张量索引、实际值、参考值、最大绝对误差和参与 rank。

## 现有 Qwen3 回放测试（8 卡）

```bash
VIME_TEST_CHECK_PARAMETER_GRADS=1 \
VIME_TEST_GRAD_RTOL=0.01 VIME_TEST_GRAD_ATOL=1e-6 \
python tests/test_qwen3_0.6B_parallel_check.py
```

两种 loss reduction 继续使用同一份 rollout 回放；基准与各并行布局额外保存梯度。
`parallel-grads-*` 临时目录保留供排查。逐元素判断公式为
`abs(actual-reference) <= atol + rtol * abs(reference)`。
需根据精度和后端设置容差；默认值不代表所有 BF16 kernel 和布局均已满足该阈值。
默认开启 GPU CI 前，须在标准训练镜像验证完整 8 卡回放。

可离线重新比较，不必重跑训练：

```bash
python -m vime.backends.megatron_utils.gradient_check /path/to/reference /path/to/replay \
  --rtol 0.01 --atol 1e-6
```

## 真实 Megatron 小模型验证（1–2 卡）

使用 `docker/Dockerfile` 固定的 Megatron 版本。模型为 4 层 GPT，hidden size 64、
词表 128、GQA、gated MLP；逻辑权重、token 固定，mask 长度不同，累积 4 个样本。
测试执行真实 Megatron pipeline schedule、梯度缓冲区、NCCL 和 distributed Adam，
attention 使用本地 PyTorch 后端，无需下载模型或数据集。在仓库根目录运行，
设置 `PYTHONPATH=.`；若 Megatron 未安装，同时加入其源码目录。

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tests/gradient_check_megatron.py --out /tmp/grad-base
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/gradient_check_megatron.py --tp 2 --out /tmp/grad-tp2
python -m vime.backends.megatron_utils.gradient_check /tmp/grad-base/grads /tmp/grad-tp2/grads \
  --rtol 1e-3 --atol 1e-6
```

省略 `--tp 2` 可运行 DP=2，替换为 `--pp 2` 可运行 PP=2。
`--overlap` 验证多 gradient bucket；`--per-token-loss` 验证 token 归一化；
`--bf16` 使用 BF16。只比较精度、loss 模式相同的运行。
脚本另存 selected-token logprob 和 Adam 更新后参数，并检查重复保存不修改梯度。
这属于诊断集成测试，不是完整 RL rollout 或收敛验证。BF16 TP 的累积顺序差异可能
需要更大的梯度绝对容差；不能由梯度接近直接推断 optimizer update 接近。

## 支持范围

支持 dense BF16/FP32 Megatron DDP、非共享 embedding/output 权重、单 distributed
optimizer instance。FP16 loss scaling、FP8、MoE/EP、共享权重和多个 optimizer
instance 会明确拒绝。CP 使用 buffer 的 DP-with-CP group，但本地 attention 小模型
测试未覆盖 SP 或 CP attention；须通过标准镜像的完整回放验证 SP/CP。
本补丁没有新增生产环境参数更新或 selected-token logprob 检查开关。

CPU 回归测试仅需 PyTorch 和 pytest，无需 Megatron：

```bash
python -m pytest tests/test_gradient_check.py
```
