# Sparse HCCL 专项回归

本目录集中维护 sparse HCCL 的差分、聚合、加载、生命周期与 CI 配置回归。文件均使用 `test_` 前缀。

`tests/test_qwen3_30B_A3B_sparse_hccl_npu.py` 保留原路径、内容和 Buildkite 注册；共享的 updater 工厂测试仍在 `tests/test_update_weight_factory.py`。

## 单进程回归

在仓库根目录、有 VIME/Megatron/vLLM-Ascend 依赖的环境运行：

```bash
python -m pytest -q tests/sparse_hccl \
  --ignore=tests/sparse_hccl/test_npu_sparse_gather_regression.py \
  --ignore=tests/sparse_hccl/test_bridge_delta_tp_differential.py \
  tests/test_update_weight_factory.py
```

这两个被排除的文件是独立分布式脚本，必须用 torchrun，不能通过普通 pytest 当作通信测试执行。

## 真实四卡 HCCL 聚合

准备 CANN 和依赖环境后，在四张空闲 NPU 上运行：

```bash
torchrun --nproc-per-node=4 tests/sparse_hccl/test_npu_sparse_gather_regression.py
```

直接导入当前仓库实现，不需要 `SPARSE_GATHER_CANDIDATE`。覆盖 FP32/BF16、空载荷、不等长、非连续输入、分块和暂存区复用，共 20 组场景。

## Bridge 全量导出对照

```bash
MODEL_KIND=qwen3_moe TP_SIZE=2 PP_SIZE=2 EP_SIZE=2 \
  torchrun --nproc-per-node=4 tests/sparse_hccl/test_bridge_delta_tp_differential.py
```

该脚本依赖 Megatron-Bridge 和对应加速器环境；比较小模型的差分重建与真实全量导出，不代替 30B 端到端看护。
