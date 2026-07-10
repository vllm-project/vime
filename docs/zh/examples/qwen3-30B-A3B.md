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

1. 为了支持在 8xH100 环境中运行 Qwen3-30B-A3B，我们需要开启 megatron 的 CPU Adam 以节省显存，对应配置为：

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

   类似地，如果要在 attention 上做 DP 同时在 expert 上做 EP，可以加
   `--vllm-data-parallel-size N` 配合 `--vllm-enable-expert-parallel`。

### bf16 训练 fp8 推理

vime 也支持 bf16 训练、fp8 推理。对于 Qwen3-30B-A3B 模型，只需额外下载 fp8 权重：

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8
```

并将脚本中的 `--hf-checkpoint` 替换为：

```bash
#--hf-checkpoint /root/Qwen3-30B-A3B
--hf-checkpoint /root/Qwen3-30B-A3B-FP8
```

即可触发 fp8 推理。目前我们会将 bf16 权重直接 cast 为 fp8，后续会逐渐加入对精度影响更小的量化方案。

⚠️  训练用的 megatron checkpoint 仍需是最初用 bf16 huggingface 权重转换得到的（`--ref-load` / `--load` 不变）。

### 多机支持

以下以 **2 节点 × 8 卡（共 16 GPU）colocate** 为例。多机与单机的差异只在于"跨节点启动 Ray"和"调整几个资源/并行度参数"，训练脚本主体不变。

1. **共享存储**：模型、数据、checkpoint 放在所有节点路径一致且都能访问的位置（如 NFS）。

2. **跨节点启动 Ray**（在训练脚本之外，各节点手动执行；详见 [快速开始 — 多机训练](../get_started/quick_start.md#大规模-moe-模型的多机训练)）：

   ```bash
   # Head 节点（node0）；MASTER_ADDR 用局域网 IP，不能是 127.0.0.1
   export MASTER_ADDR=<head 局域网 IP>
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

   # 其余各节点
   ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
   ```

   等 `ray status` 显示 16 GPU 后再提交。由于集群是你手动起好的，运行前需让脚本跳过它开头的进程管理逻辑——把开头的**清理块**（`ray stop --force`、`pkill -9 ray`、`pkill -9 python`、`pkill -9 redis`）**和** `ray start --head ...` 一行都删掉（或注释掉）。否则脚本会先把你刚起的 head 杀掉（并让 worker 变孤儿），导致 `ray job submit` 连 `http://127.0.0.1:8265` 失败。脚本其余部分保留——它仍会 source 模型参数并执行 `ray job submit`。

3. **调整脚本参数**（`scripts/run-qwen3-30B-A3B.sh`）：
   - 把 `train.py` 的 `--actor-num-nodes` 由 `1` 改为 `2`（`--actor-num-gpus-per-node` 保持 8）。colocate 下 `--rollout-num-gpus` 会自动取 `actor_num_gpus_per_node × actor_num_nodes = 16`，无需手设。
   - 卡数翻倍后相应增大 `PERF_ARGS` 的并行度（如提高 TP 或引入 DP）；大规模的具体配比可参考 [GLM-4.7](glm4.7-355B-A32B.md)、[DeepSeek-R1](deepseek-r1.md) 等更大集群的例子。
   - `global-batch-size` 必须等于 `rollout-batch-size × n-samples-per-prompt`。
   - （可选）多机使用 distributed optimizer，optimizer 显存压力下降，可去掉 `OPTIMIZER_ARGS` 中的 CPU Adam（`--optimizer-cpu-offload` 等）以提速。

4. **让每个 vLLM engine 留在单节点内**：推荐 `--rollout-num-gpus-per-engine 8`（每节点 1 个 engine），而不是 `16`（单个 engine 跨 2 节点 TP=16）。跨节点 TP 会明显变慢、且对 per-token 数值更敏感；该值需整除总推理卡数（此例为 16）。

⚠️  常见问题：
- **Worker 加不进 Ray / NCCL 失败**：检查 `MASTER_ADDR`、容器 `/etc/hosts`（hostname 勿指向 `127.0.0.1`）、多网卡时设 `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`。
- **`Not enough samples X for global_batch_size Y`**：保持 `global-batch-size = rollout-batch-size × n-samples-per-prompt`。
- **每节点少于 8 卡（colocate）**：需显式设置 `--num-gpus-per-node`。

此外，当总卡数不能被 expert 总数整除时，可以开启 vLLM 的 EPLB（Expert Parallelism Load Balancer），通过 `--vllm-eplb-config` 配置冗余 expert。例如 24 卡的场景：

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
