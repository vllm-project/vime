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

### bf16 训练 fp8 推理

vime 还支持 bf16 训练，fp8 推理。对于 Qwen3-30B-A3B 模型，只需要下载如下模型：

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8
```

并将 `--hf-checkpoint` 替换为：

```bash
#--hf-checkpoint /root/Qwen3-30B-A3B
--hf-checkpoint /root/Qwen3-30B-A3B-FP8
```

即可触发 fp8 训练。目前我们会将 bf16 权重直接 cast 为 fp8，后续会逐渐添加对精度影响更小的量化方案。

⚠️  训练的 megatron checkpoint 还需要是最开始用 bf16 的 huggingface 转换的。

### 多机支持

对于多机环境，需要进行如下的几点修改：
- 将训练模型，数据放在所有机器都可以访问到的路径上；
- 设置各台机器都可以访问到的 `MASTER_ADDR` 之外；
- 去掉 CPU adam 相关的配置，因为使用了 distributed optimizer，所以多机环境下 optimizer 的显存占比会明显下降。

除此之外，还可以进行如下的修改：

- 当总卡数并不能被 expert 总数整除时，可以开启 vLLM 的 EPLB（Expert Parallelism Load Balancer），通过 `--vllm-eplb-config` 配置冗余 expert。例如对于 24 卡的场景：

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
