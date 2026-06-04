# 8xH100 训练 Qwen3-30B-A3B

## 环境准备

搭建环境、下载模型、数据与 ckpt 转换均与 Qwen3-4B 模型相同，可以参考 [示例：Qwen3-4B](qwen3-4B.md)，将文中 Qwen3-4B 的部分转换为 Qwen3-30B-A3B 即可。

可以用如下方法把 huggingface checkpoint 转化为 torch_dist 格式：

```bash
cd vime/
pip install -e . --no-deps
source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-30B-A3B/ \
   --save /root/Qwen3-30B-A3B_torch_dist/
```

## 执行训练

执行训练：

```bash
cd /root/vime
bash scripts/run-qwen3-30B-A3B.sh
```

### 参数简介

这里我们简单介绍一下脚本 [run-qwen3-30B-A3B.sh](https://github.com/vllm-project/vime/blob/main/scripts/run-qwen3-30B-A3B.sh) 中与 MoE 相关的部分。

1. 为了支持在 8xH800 环境中运行 Qwen3-30B-A3B，我们需要开启 megatron 的 CPU Adam 以节省显存，对应配置为：

   ```bash
   OPTIMIZER_ARGS=(
      ...
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
   ```

2. 开启 megatron 支持的 moe 优化，当前配置为 tp4, ep8：

   ```bash
   PERF_ARGS=(
      --tensor-model-parallel-size 4
      --sequence-parallel
      --pipeline-model-parallel-size 1
      --context-parallel-size 1
      --expert-model-parallel-size 8
      --expert-tensor-parallel-size 1
      ...
   )
   ```

3. 在 vLLM 侧开启 MoE expert parallelism。vLLM 中 EP size 由
   `tensor_parallel_size × data_parallel_size` 自动推导，所以 8 卡 engine 只需
   `--vllm-enable-expert-parallel` 就是 EP=8：

   ```bash
   VLLM_ARGS=(
      --rollout-num-gpus-per-engine 8
      --vllm-gpu-memory-utilization 0.7
      --vllm-enable-expert-parallel
      --vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256)
   )
   ```

   如果要在 attention 上做 DP 同时在 expert 上做 EP，可以加 `--vllm-data-parallel-size N`
   配合 `--vllm-enable-expert-parallel`。

### 多机支持

以下以 **2台机器、每台8卡（共16GPU）** 为入门示例；脚本与参数可扩展到 **N节点**。多机与单机的主要差异：

- 训练模型、数据放在所有节点均可访问的路径（如 NFS）；
- `MASTER_ADDR` 设为 head 节点的 **局域网 IP**（非 `127.0.0.1`）；
- 去掉 CPU Adam（多机使用 distributed optimizer，无需 `--optimizer-cpu-offload`）；
- `global-batch-size` 必须等于 `rollout-batch-size × n-samples-per-prompt`。

#### 拓扑概览

| 组件 | 双机默认配置 |
|------|------|
| 集群 | `ACTOR_NUM_NODES=2`，`ACTOR_NUM_GPUS_PER_NODE=8` |
| Megatron训练 | TP=8, EP=8, CP=2（expert 分片跨节点） |
| vLLM Rollout | 跨节点 TP=16（`rollout-num-gpus-per-engine = 节点数 × 每节点GPU`） |
| 调度 | Ray 集群 + `--colocate` 共卡模式 |

转换 checkpoint 时建议使用与训练一致的 Megatron 并行度（双机示例 TP=8, EP=8）。checkpoint 的 EP 需与 `--expert-model-parallel-size` 一致，否则 `load_checkpoint` 可能极慢或卡住。

#### 启动 Ray 集群

Ray 集群需在各节点上 **单独启动**，不在训练脚本内。先在所有 worker 节点加入集群，确认 `ray status` 显示预期 GPU 总数后，再在 head 提交训练。示例（双机）：

```bash
# === Head 节点 ===
export MASTER_ADDR=<head_局域网_IP>
ray start --head --node-ip-address="${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

# === 各 Worker 节点 ===
export MASTER_ADDR=<head_局域网_IP>
ray start --address="${MASTER_ADDR}:6379" --node-ip-address=<本机_局域网_IP> --num-gpus 8
```

更多说明见 [快速开始 — 多机训练](../get_started/quick_start.md#multi-node-training-for-large-scale-moe-models)。

#### 执行训练

Ray 集群就绪后，在 **head 节点** 设置多机环境变量并运行 **与单机相同的脚本**（`ACTOR_NUM_NODES>1` 时脚本不会启动 Ray，并使用多机默认参数）：

```bash
export MASTER_ADDR=<head_局域网_IP>
export ACTOR_NUM_NODES=2
export ACTOR_NUM_GPUS_PER_NODE=8
cd /root/vime
bash scripts/run-qwen3-30B-A3B.sh
```

2 step 冒烟示例：

```bash
NUM_ROLLOUT=2 ENABLE_R3=0 bash scripts/run-qwen3-30B-A3B.sh
```

扩展到 N 节点（例如 4×8）时，在各 worker 加入 Ray 后，于 head 设置 `ACTOR_NUM_NODES=4` 并按总卡数调整 `MEGATRON_TP` / `MEGATRON_EP` / `MEGATRON_CP` / `ROLLOUT_NUM_GPUS_PER_ENGINE`。

#### 多机关键参数

| 变量 | 双机默认 | 说明 |
|------|----------|------|
| `ACTOR_NUM_NODES` | 2（单机默认为 1） | 训练节点总数（含 head）；>1 时脚本不启动 Ray |
| `ACTOR_NUM_GPUS_PER_NODE` | 8 | 每节点 GPU 数 |
| `MEGATRON_TP` / `MEGATRON_EP` / `MEGATRON_CP` | 8 / 8 / 2 | Megatron 并行 |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | 总 GPU 数 | vLLM engine 占用卡数 |
| `ENABLE_R3` | 1 | 设为 0 可关闭 R3 路径 |

脚本默认 batch：`rollout-batch-size=4`，`n-samples-per-prompt=2`，`global-batch-size=8`；vLLM 使用 `--vllm-moe-backend triton`。

#### 多机常见问题

- **Worker 无法加入 Ray / NCCL 失败**：检查 `MASTER_ADDR`、容器 `/etc/hosts`（hostname 勿指向 `127.0.0.1`）、`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`。
- **`Not enough samples X for global_batch_size Y`**：同步调整 `global-batch-size` 与 `rollout-batch-size × n-samples-per-prompt`。
- **GPU 显存占满但无进程**：重启容器或 `ray stop --force` 清理残留 vLLM 上下文。

#### EPLB

当总卡数并不能被 expert 总数整除时，可以开启 vLLM 的 EPLB（Expert Parallelism Load Balancer），通过 `--vllm-eplb-config` 配置冗余 expert。例如对于 24 卡的场景：

   ```bash
   VLLM_ARGS=(
      --rollout-num-gpus-per-engine 24
      --vllm-gpu-memory-utilization 0.7
      --vllm-data-parallel-size 3
      --vllm-enable-expert-parallel
      --vllm-enable-eplb
      --vllm-eplb-config '{"num_redundant_experts": 16}'
   )
   ```
