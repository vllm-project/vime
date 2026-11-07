# Speculative Decoding

Speculative decoding is a key optimization for speeding up rollouts. Instead of having the expensive target model decode token by token during inference, a lightweight draft model first decodes ahead to produce several tokens, and then the target model verifies them in a batch.

## Accelerating Inference with Speculative Decoding

vLLM exposes speculative decoding as a single JSON config (`SpeculativeConfig`),
which vime forwards via `--vllm-speculative-config`. For models with MTP layers
(e.g., GLM-4.7, DeepSeek-V3/R1), pass:

```bash
--vllm-speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

To use a separately trained draft model, set `model` (and optionally `draft_tensor_parallel_size`)
in the same JSON:

```bash
--vllm-speculative-config '{"method":"eagle","num_speculative_tokens":4,"model":"/your/draft/model/path"}'
```

To train a draft model from scratch, see [TorchSpec](https://github.com/lightseekorg/TorchSpec)
and [vllm-project/speculators](https://github.com/vllm-project/speculators).
TorchSpec provides torch-native, disaggregated draft training.
Speculators supports EAGLE-3, DFlash, and MTP-style drafts, ships pre-trained
checkpoints on Hugging Face (see the `RedHatAI/*-speculator.*` collection), and
saves drafts in a format that `vllm serve <speculator_model>` can deploy directly.

For the full list of `SpeculativeConfig` fields (including `num_speculative_tokens`
and `draft_tensor_parallel_size`), see vLLM's speculative-decoding
[documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

## Configure EAGLE in the interactive builder

In the homepage's **Serving** step, enable **EAGLE · Speculative decoding**. Choose the checkpoint's built-in MTP head where supported, or **Separate EAGLE head** and enter its checkpoint directory or Hugging Face model ID. The separate head must be compatible with the target model; an arbitrary smaller language model is not an EAGLE head. Make its weights accessible to every serving host.

The builder exports one vLLM `SpeculativeConfig`: `method="mtp"` for a supported built-in head or `method="eagle"` with `model` for a separate EAGLE head. Drafting depth $\gamma$ maps to `num_speculative_tokens=γ`. The default depth is 3 (4 for the GLM-5.3 recipe). These are starting settings, not measured optimal values. Attention backend selection follows the model and vLLM configuration.

The **Your little RL world** diagram shows all eight PD / HiCache / EAGLE combinations. PD separates prefill and decode engine pools and adds cache transfer. HiCache adds a CPU cache tier; in this recipe, PD attaches it only to prefill. EAGLE adds a draft head, candidate tokens, target verification and committed output. With EAGLE off, the target decodes one token per step. **Play a round** highlights these stages in order; the acceptance example is illustrative.

Managed deployments receive `--vllm-speculative-config` arguments in `experiment.sh`. External deployments must apply the corresponding `--speculative-config` arguments from `serving-reference.sh` when launching their engines. Training-side flags also enable vime's speculation metrics; they cannot reconfigure a running external engine. The draft path is retained locally and in the downloaded configuration, but omitted from share links. Switching target models clears it so a head is not accidentally reused for another target.

The builder configures inference. It does not enable online MTP or separate-head training. As the target changes during RL, monitor `spec_accept_rate`, `spec_accept_length` and rollout latency. Extra draft weights, verification buffers and hybrid recurrent-state snapshots also consume memory.

## Why verification is needed

[EAGLE](https://arxiv.org/abs/2401.15077) uses a lightweight head to propose candidates using target-model features. The target then verifies them. The probability argument below is the standard single-path rejection-sampling construction from [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318); it describes the ideal sampler, not a claim that every backend uses identical kernels.

At a fixed prefix, let $p(a)$ be the target distribution and $q(a)$ the draft distribution. For a proposed $a\sim q$, accept with probability

$$\alpha(a)=\min\left(1,\frac{p(a)}{q(a)}\right).$$

The accepted probability mass for token $a$ is $q(a)\alpha(a)=\min(p(a),q(a))$. Total rejection probability is

$$Z=1-\sum_a\min(p(a),q(a))=\sum_a[p(a)-q(a)]_+.$$

After rejection, draw a correction from $r(a)=[p(a)-q(a)]_+/Z$. Combining accepted and corrected mass gives

$$\Pr(\text{output}=a)=\min(p(a),q(a))+Zr(a)=p(a).$$

When $Z=0$, rejection never occurs. Apply the construction at successive accepted prefixes; after the first rejection, discard the remaining candidates and continue from the corrected prefix. If all candidates pass, an additional target token can be emitted. Thus the diagram marks an accepted prefix and a correction, rather than training directly on unchecked draft tokens.

For a simple cost model, let $A$ be the number of accepted draft tokens, $t_d$ the per-step draft cost, $t_v$ the batched target verification cost, and $t_o$ cache/scheduling overhead. If an ordinary target step costs $t_t$,

$$\text{speedup}\approx\frac{(\mathbb E[A]+1)t_t}{\gamma t_d+t_v+t_o}.$$

More draft steps help only when the additional accepted tokens repay the extra work. This is why the builder exposes depth and acceptance metrics rather than promising a fixed speedup.

## Online SFT for the Draft Model

As RL progresses, the sampling distributions of the draft and target models can drift apart. Fewer draft tokens pass verification, and speculative decoding can even yield negative returns.

vime currently supports online training of the MTP layers during RL, updating the draft model in sync with training to consistently improve sampling speed. See the related rationale in this [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e). Use it as follows:

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

And note that this requires a torch dist checkpoint with the MTP weight, you need to add `--mtp-num-layers 1` during the checkpoint conversion from huggingface to torch dist.

Training external draft models is still a WIP.
