# 可复现性

可复现性是科研进展的基础。vime 通过结合 vLLM 的 [batch-invariant 确定性推理](https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference) 与 Megatron-LM 的 deterministic 模式，支持 bitwise 级的实验复现。

为了开启确定性训练，你需要通过 `pip uninstall flash_attn_3 -y` 卸载 flash attention 3，并设置：

```bash
  # vLLM config
  --vllm-enable-deterministic-inference
  --vllm-attention-backend flashinfer

  # megatron config
  --deterministic-mode
```

以及设置如下环境变量：

```bash
     "env_vars": {
        ...,
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
```

我们提供了一个完全确定性的，用 Qwen2.5 0.5B 训练 GSM8K 的脚本。

可以用如下脚本初始化训练数据和 ckpt：

```bash
# download
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# convert ckpt
cd vime/
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM/ python \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
   --save /root/Qwen2.5-0.5B-Instruct_torch_dist/
```

可以使用如下脚本进行训练：

```bash
bash scripts/run-qwen2.5-0.5B-reproducibility.sh
```

这个 PR 中记录了 wandb 的截图 [pull#370](https://github.com/THUDM/slime/pull/370)。

## Train/rollout log-prob alignment（GLM-5）

除单侧 bitwise 复现外，vime 还可以对齐训练与 rollout（推理）的 log-prob。目前该能力只支持 **GLM-5 结构**（MLA + DSA sparse attention），并要求 deterministic VLLM、batch-invariant DeepGEMM 与 DeepEP 构建。所需 Megatron 侧对齐 hook 由 Vime 在运行时安装，不需要额外 Megatron patch。

Supported in this path:

- DSA sparse attention (`flashmla_sparse` prefill/decode), including deterministic NSA RadixCache/prefix cache;
- DeepGEMM batch-invariant block-FP8 forward for dense and grouped-MoE layers (with BF16 backward);
- fp32 MoE router (the LM head stays bf16 on both train and rollout — matching precision, not fp32, is what aligns);
- VLLM rollout 使用 DeepEP low-latency，Megatron 训练使用 DeepEP normal。
  第二次小 payload normal dispatch 保留每个 top-k route，token owner 按
  slot 顺序做 FP32 加权归约；这条对齐路径不支持普通 Megatron all-to-all；
- 支持 bf16 或 FP8-E4M3 KV cache。`flashmla_sparse` 路径把 KV 以 FP8
  packed 格式保存，只 gather 并反量化被选中的 page，再交给 BF16 sparse
  kernel。维护的 gate 默认使用 FP8-E4M3，不使用 rollout routing replay
  (R3)，因此包括 router 和 experts 在内的主模型参数都会执行 backward；
  辅助 DSA indexer 通过 `--freeze-indexer` 始终保持冻结。

回归 gate 是 `tests/test_glm52_6layer_deterministic_e2e.py`（6-layer GLM-5.2，
单机 EP8）：它执行真实的 Megatron→VLLM online-weight-update rollout，
训练全部主模型参数，并断言 `train_rollout_logprob_abs_diff < 1e-6`（已验证的
DeepEP 对齐参考结果为 `x e-7` 量级）。

另有一个较短的 EP8 gate `tests/test_glm52_layerwise_zero_e2e.py`，会同时
记录训推两侧 decoder layer 0–5 的可见输出，并要求所有匹配 hidden-state
元素的绝对误差严格等于 0。
