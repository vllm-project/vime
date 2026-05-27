# 投机采样

投机采样是加速 rollout 的重要优化手段。推理过程中不再让昂贵的 Target Model 逐个 token 进行 decode，而是先由一个轻量级的 draft model 先进行 decode，生成多个 token 后，再由大模型进行批量验证。

## 使用投机采样加速推理

vLLM 把投机采样的所有配置收敛到一个 JSON（`SpeculativeConfig`），slime 通过
`--vllm-speculative-config` 透传。对于有 MTP 层的模型（例如 GLM-4.7、DeepSeek-V3/R1），传入：

```bash
--vllm-speculative-config '{"method":"eagle","num_speculative_tokens":3}'
```

如果要使用单独训练的 draft model，在同一个 JSON 里加上 `model`（可选还可加
`draft_tensor_parallel_size` 等）：

```bash
--vllm-speculative-config '{"method":"eagle","num_speculative_tokens":3,"model":"/your/draft/model/path"}'
```

要从头训练一个 draft model，可以参考 [TorchSpec](https://github.com/lightseekorg/TorchSpec)
和 [vllm-project/speculators](https://github.com/vllm-project/speculators)。
TorchSpec 提供 torch-native 的 disaggregated draft training。
Speculators 支持 EAGLE-3、DFlash 以及 MTP 风格的 draft，HuggingFace 上已有预训练 ckpt
（参见 `RedHatAI/*-speculator.*` 集合），产物可被 `vllm serve <speculator_model>` 直接部署。

`SpeculativeConfig` 的完整字段（`disable_by_batch_size`、`acceptance_method`、
draft TP 等）请参考 vLLM 的 speculative decoding [文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

## 在线 SFT draft model

随着 RL 流程的进行，draft model 和 target model 的采样概率差异逐渐增大，能通过验证的 draft token 逐渐减少，spec 甚至可能造成负收益。

目前，slime 支持了在 RL 流程中在线训练 MTP 层，随着训练的进行同步更新 draft model，稳定提高了采样速度，相关原理可参见 [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e)。使用方法如下：

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

注意 MTP 训练需要一个包含了 MTP 权重的 checkpoint，所以在将 huggingface checkpoint 转为 torch dist 时，也需要加上 `--mtp-num-layers 1`。

外部 draft model 的训练还在 WIP。
