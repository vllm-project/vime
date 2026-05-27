# Speculative Decoding

Speculative decoding is a key optimization for speeding up rollouts. Instead of having the expensive target model decode token by token during inference, a lightweight draft model first decodes ahead to produce several tokens, and then the target model verifies them in a batch.

## Accelerating Inference with Speculative Decoding

vLLM exposes speculative decoding as a single JSON config (`SpeculativeConfig`),
which slime forwards via `--vllm-speculative-config`. For models with MTP layers
(e.g., GLM-4.7, DeepSeek-V3/R1), pass:

```bash
--vllm-speculative-config '{"method":"eagle","num_speculative_tokens":3}'
```

To use a separately trained draft model, set `model` (and optionally `draft_tensor_parallel_size`)
in the same JSON:

```bash
--vllm-speculative-config '{"method":"eagle","num_speculative_tokens":3,"model":"/your/draft/model/path"}'
```

To train a draft model from scratch, see [TorchSpec](https://github.com/lightseekorg/TorchSpec)
and [vllm-project/speculators](https://github.com/vllm-project/speculators).
TorchSpec provides torch-native, disaggregated draft training.
Speculators supports EAGLE-3, DFlash, and MTP-style drafts, ships pre-trained
checkpoints on Hugging Face (see the `RedHatAI/*-speculator.*` collection), and
saves drafts in a format that `vllm serve <speculator_model>` can deploy directly.

For the full list of `SpeculativeConfig` fields (including `disable_by_batch_size`,
`acceptance_method`, draft TP, etc.), see vLLM's speculative-decoding
[documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

## Online SFT for the Draft Model

As RL progresses, the sampling distributions of the draft and target models can drift apart. Fewer draft tokens pass verification, and speculative decoding can even yield negative returns.

slime currently supports online training of the MTP layers during RL, updating the draft model in sync with training to consistently improve sampling speed. See the related rationale in this [blog](https://www.notion.so/jiajunli-guapisolo/Power-Up-Speculative-Decoding-In-Reinforcement-Learning-2a92d24a293b802d9c73dbae429e581e). Use it as follows:

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

And note that this requires a torch dist checkpoint with the MTP weight, you need to add `--mtp-num-layers 1` during the checkpoint conversion from huggingface to torch dist.

Training external draft models is still a WIP.
