"""The policy replays each input's noise while the value model trains its core."""

import copy

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

pytest.importorskip("megatron")
pytest.importorskip("vllm_rlt")
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt import LLM, CacheConfig, SamplingParams
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM

from vime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from vime.utils.types import RecurrentTrace
from vime_plugins.huginn.model import HuginnMegatronModel


def trace(seed):
    return RecurrentTrace(
        schema_version=1,
        model_family="huginn_raven",
        model_revision="tiny-cpu",
        engine_revision="test",
        runtime_epoch="cpu",
        policy_version=0,
        publication_digest="test",
        request_id=str(seed),
        seed=seed,
        prefill_depth=2,
        decode_depths=[2, 2],
        temperature=0.8,
        finish_reason="length",
        latent_seed=seed,
        latent_profile="like-init-cpu-f32-v1",
    )


@pytest.mark.parametrize("recompute", [False, True])
def test_packed_policy_values_gradients_and_hf_export(recompute, tmp_path):
    torch.manual_seed(42)
    native = HuginnForCausalLM(
        HuginnConfig(
            n_embd=8,
            n_heads=2,
            n_layers=3,
            n_layers_in_prelude=1,
            n_layers_in_recurrent_block=1,
            n_layers_in_coda=1,
            intermediate_size=16,
            mean_recurrence=2,
            block_size=32,
            vocab_size=11,
            padded_vocab_size=11,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )
    )
    config = TransformerConfig(num_layers=3, hidden_size=8, num_attention_heads=2, ffn_hidden_size=16)
    actor = HuginnMegatronModel(config, copy.deepcopy(native), recompute=recompute, model_revision="tiny-cpu")
    critic = HuginnMegatronModel(config, copy.deepcopy(native), role="critic", recompute=recompute)
    engine = LLM(native, cache_config=CacheConfig(num_blocks=8))
    prompts, seeds = [[1, 2], [3]], [7, 8]
    outputs = engine.generate(
        prompts,
        [
            SamplingParams(
                max_tokens=2,
                min_loops=2,
                max_loops=2,
                seed=seed,
                latent_seed=seed,
                temperature=0.8,
                logprobs=0,
                logprobs_mode="processed",
            )
            for seed in seeds
        ],
    )
    engine.close()
    sequences = [torch.tensor(prompt + output.token_ids[:-1]) for prompt, output in zip(prompts, outputs, strict=True)]
    lengths = torch.tensor([0, len(sequences[0]), sum(map(len, sequences))], dtype=torch.int32)
    packed = PackedSeqParams(cu_seqlens_q=lengths, cu_seqlens_kv=lengths, qkv_format="thd")
    tokens = torch.cat(sequences).unsqueeze(0)
    traces = [trace(seed) for seed in seeds]
    logits = actor(tokens, packed_seq_params=packed, recurrent_inputs=traces)
    for index, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        start, end = lengths[index].item() + len(prompt) - 1, lengths[index + 1].item()
        scores = F.log_softmax(logits[0, start:end] / 0.8, -1).gather(1, torch.tensor(output.token_ids)[:, None])[:, 0]
        torch.testing.assert_close(scores, torch.tensor(output.log_probs), atol=3e-6, rtol=3e-5)
    values = critic(tokens, packed_seq_params=packed, recurrent_inputs=traces)
    assert values.shape == (*tokens.shape, 1)
    assert "lm_head.weight" not in dict(critic.named_parameters())
    (values - 1).square().mean().backward()
    F.cross_entropy(logits[0], torch.tensor([2, 3, 4, 5, 6])).backward()
    for module in (actor, critic):
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters()
        )
        reference = copy.deepcopy(module)
        reference.recompute = False
        reference.zero_grad(set_to_none=True)
        output = reference(tokens, packed_seq_params=packed, recurrent_inputs=traces)
        loss = (
            F.cross_entropy(output[0], torch.tensor([2, 3, 4, 5, 6]))
            if module.role == "actor"
            else (output - 1).square().mean()
        )
        loss.backward()
        for actual, expected in zip(module.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)

    exported = dict(
        item
        for name, parameter in actor.named_parameters()
        for item in _convert_to_hf_core(None, "huginn", name, parameter.detach())
    )
    exported["freqs_cis"] = actor.freqs_cis
    assert exported["lm_head.weight"].data_ptr() != exported["transformer.wte.weight"].data_ptr()
    save_file(exported, str(tmp_path / "model.safetensors"))
    import json

    (tmp_path / "config.json").write_text(json.dumps(native.config.to_dict()))
    restored = HuginnForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    for name, value in native.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    critic.bfloat16()
    assert critic.freqs_cis.dtype == torch.float32
    assert torch.equal(critic.freqs_cis, native.freqs_cis)
