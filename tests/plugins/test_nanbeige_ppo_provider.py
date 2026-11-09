"""Compare differentiable packed logits with the separate native KV execution."""

import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("megatron")
pytest.importorskip("vllm_rlt")
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt import LLM, CacheConfig, SamplingParams
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM

from vime_plugins.nanbeige.model import NanbeigeMegatronModel


@pytest.mark.parametrize("recompute", [False, True])
def test_two_loop_logits_values_and_gradients(recompute):
    torch.manual_seed(42)
    native = NanbeigeForCausalLM(
        NanbeigeConfig(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )
    )
    config = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2, kv_channels=8, ffn_hidden_size=16)
    actor = NanbeigeMegatronModel(config, copy.deepcopy(native), recompute=recompute)
    critic = NanbeigeMegatronModel(config, copy.deepcopy(native), role="critic", recompute=recompute)
    tokens = torch.tensor([[1, 2, 3]])
    logits = actor(tokens)
    assert actor.loop_budget == 2
    engine = LLM(native, cache_config=CacheConfig(num_blocks=8))
    outputs = engine.generate(
        [[1], [1, 2], [1, 2, 3]],
        SamplingParams(
            max_tokens=1,
            min_loops=2,
            max_loops=2,
            logprobs=0,
        ),
    )
    for position, output in enumerate(outputs):
        scores = F.log_softmax(logits[0, position].float(), dim=-1)
        assert output.token_ids[0] == int(logits[0, position].argmax())
        assert scores[output.token_ids[0]].item() == pytest.approx(output.log_probs[0], abs=2e-6)
    engine.close()
    values = critic(tokens)
    assert values.shape == (1, 3, 1)
    assert "lm_head.weight" not in dict(critic.named_parameters())
    (values - 1).square().mean().backward()
    assert all(parameter.grad is not None for parameter in critic.parameters())
    F.cross_entropy(logits[0], torch.tensor([2, 3, 4])).backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())
    for model in (actor, critic):
        reference = copy.deepcopy(model)
        reference.recompute = False
        reference.zero_grad(set_to_none=True)
        output = reference(tokens)
        loss = (
            F.cross_entropy(output[0], torch.tensor([2, 3, 4]))
            if model.role == "actor"
            else (output - 1).square().mean()
        )
        loss.backward()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
