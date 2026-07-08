# 8xH100 训练 Qwen3-30B-A3B

## 环境准备

搭建环境与 ckpt 转换均与 Qwen3-4B 相同，可参考 [示例：Qwen3-4B](qwen3-4B.md)，将文中 Qwen3-4B 替换为 Qwen3-30B-A3B 即可。

下载模型与数据：

```bash
# hf checkpoint
hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B

# 训练数据
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval 数据
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

再把 huggingface checkpoint 转化为 torch_dist 格式：

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

对于多机环境，需要进行如下几点修改：

- 将训练模型、数据放在所有机器都可以访问到的路径上（如 NFS）；
- 设置各台机器都可以访问到的 `MASTER_ADDR`（非 `127.0.0.1`），并在各节点手动启动 Ray 后再从 head 提交训练（见 [快速开始 — 多机训练](../get_started/quick_start.md#大规模-moe-模型的多机训练)）；
- 按总卡数相应调整 `train.py` 的 `--actor-num-nodes` 以及 `PERF_ARGS` 中的并行度（TP/EP/CP）；
- 去掉 CPU adam 相关配置，因为多机使用 distributed optimizer，optimizer 的显存占比会明显下降。

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
