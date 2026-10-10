# 投机采样

投机采样是加速 rollout 的重要优化手段。推理过程中不再让昂贵的 Target Model 逐个 token 进行 decode，而是先由一个轻量级的 draft model 先进行 decode，生成多个 token 后，再由大模型进行批量验证。

## 使用投机采样加速推理

vLLM 把投机采样的所有配置收敛到一个 JSON（`SpeculativeConfig`），vime 通过
`--vllm-speculative-config` 透传。对于有 MTP 层的模型（例如 GLM-4.7、DeepSeek-V3/R1），传入：

```bash
--vllm-speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

如果要使用单独训练的 draft model，在同一个 JSON 里加上 `model`（可选还可加
`draft_tensor_parallel_size` 等）：

```bash
--vllm-speculative-config '{"method":"eagle","num_speculative_tokens":4,"model":"/your/draft/model/path"}'
```

要从头训练一个 draft model，可以参考 [TorchSpec](https://github.com/lightseekorg/TorchSpec)
和 [vllm-project/speculators](https://github.com/vllm-project/speculators)。
TorchSpec 提供 torch-native 的 disaggregated draft training。
Speculators 支持 EAGLE-3、DFlash 以及 MTP 风格的 draft，HuggingFace 上已有预训练 ckpt
（参见 `RedHatAI/*-speculator.*` 集合），产物可被 `vllm serve <speculator_model>` 直接部署。

`SpeculativeConfig` 的完整字段（`num_speculative_tokens`、`draft_tensor_parallel_size`、
draft TP 等）请参考 vLLM 的 speculative decoding [文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

## 在交互向导中配置 EAGLE

在首页 **Serving / 推理引擎** 步骤开启 **EAGLE · 投机采样**。有内置 MTP 配方的模型可以使用 checkpoint 自带的预测层；也可选择「独立 EAGLE 头」，填写 checkpoint 目录或 Hugging Face 模型 ID。独立头必须匹配目标模型，任意较小的语言模型并不等于 EAGLE 头；每台推理主机都需要能访问其权重。

向导导出一个 vLLM `SpeculativeConfig`：支持的内置投机头使用 `method="mtp"`，独立 EAGLE 头使用 `method="eagle"` 和 `model`。投机深度 $\gamma$ 映射到 `num_speculative_tokens=γ`，默认 3（GLM-5.3 配方为 4）。这些是起点设置，不是实测最优值；attention backend 按模型与 vLLM 配置选择。

**Your little RL world** 展示 PD / HiCache / EAGLE 的八种开关组合。PD 把 prefill、decode 分成两个引擎池，增加缓存传输；HiCache 增加 CPU 缓存层，当前 PD 配方只在 prefill 使用它；EAGLE 增加投机头、候选 token、目标模型验证和最终提交。关闭 EAGLE 时，目标模型逐 token 解码。「演示一轮」会依次高亮这些阶段；接受与拒绝的结果仅为示意。

托管引擎的 `experiment.sh` 使用 `--vllm-speculative-config` 参数。外部引擎需要在启动时应用 `serving-reference.sh` 中的 `--speculative-config` 参数；训练侧参数还负责启用 vime 的投机采样指标，但不能重新配置已运行的外部引擎。投机头路径保存在本地和下载的配置中，分享链接会移除该路径。切换目标模型会清空路径，避免误用另一模型的投机头。

向导配置的是推理，不会开启在线 MTP 或独立投机头训练。随着 RL 改变目标模型，需要观察 `spec_accept_rate`、`spec_accept_length` 和 rollout 延迟。投机头权重、验证 buffers，以及 hybrid 递归状态快照也会占用显存。

## 为什么需要验证

[EAGLE](https://arxiv.org/abs/2401.15077) 用轻量投机头结合目标模型特征提出候选，再交给目标模型验证。下面是 [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) 中标准单路径拒绝采样的概率推导，描述理想采样过程，不意味着各后端的 kernel 完全相同。

固定一个 prefix，设 $p(a)$ 为目标分布，$q(a)$ 为 draft 分布。对 $a\sim q$，接受概率为：

$$\alpha(a)=\min\left(1,\frac{p(a)}{q(a)}\right).$$

Token $a$ 被接受的概率质量是 $q(a)\alpha(a)=\min(p(a),q(a))$。总拒绝概率为：

$$Z=1-\sum_a\min(p(a),q(a))=\sum_a[p(a)-q(a)]_+.$$

拒绝后，从残差分布 $r(a)=[p(a)-q(a)]_+/Z$ 采样修正 token。合并接受与修正的概率质量：

$$\Pr(\text{output}=a)=\min(p(a),q(a))+Zr(a)=p(a).$$

当 $Z=0$ 时不会拒绝。在逐步接受的 prefix 上重复这一过程；首次拒绝后丢弃其余候选，从修正后的前缀继续。若所有候选都被接受，还可多输出一个目标 token。因此图中展示的是接受前缀与修正 token，不会直接拿未经验证的 draft token 去训练。

用一个简化模型分析收益：$A$ 为接受的候选数，$t_d$ 为单步 draft 耗时，$t_v$ 为目标模型批量验证耗时，$t_o$ 为缓存和调度开销，普通目标模型单步解码耗时为 $t_t$，则：

$$\text{speedup}\approx\frac{(\mathbb E[A]+1)t_t}{\gamma t_d+t_v+t_o}.$$

增加投机步数只有在额外接受的 token 能抵消额外计算时才有收益。因此向导提供深度与接受率指标，不承诺固定加速倍数。

## 在线 SFT draft model

随着 RL 流程的进行，draft model 和 target model 的采样概率差异逐渐增大，能通过验证的 draft token 逐渐减少，spec 甚至可能造成负收益。

目前，vime 支持了在 RL 流程中在线训练 MTP 层，随着训练的进行同步更新 draft model，稳定提高了采样速度，相关原理可参见 [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e)。使用方法如下：

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

注意 MTP 训练需要一个包含了 MTP 权重的 checkpoint，所以在将 huggingface checkpoint 转为 torch dist 时，也需要加上 `--mtp-num-layers 1`。

外部 draft model 的训练还在 WIP。
