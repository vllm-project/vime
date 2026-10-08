"""The recurrent critic reads hidden states and trains the shared physical core."""

import copy
import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("megatron")
pytest.importorskip("vllm_rlt")
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from vime_plugins.ouro.model import OuroMegatronModel


@pytest.mark.parametrize("family", ["ouro", "nanbeige"])
def test_role_factory_and_common_hf_loader_preserve_scalar_head(family, tmp_path, monkeypatch):
    import megatron.training
    import megatron.training.arguments
    from megatron.core import mpu
    from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM

    from vime.backends.megatron_utils.hf_to_megatron.common import load_model_hf_weights
    from vime.backends.megatron_utils.hf_to_megatron.ouro import ouro_hf_tensor
    from vime.backends.megatron_utils.model_provider import get_model_provider_func
    from vime.backends.megatron_utils.update_weight import common

    settings = dict(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
    )
    if family == "ouro":
        native = OuroForCausalLM(OuroConfig(**settings))
    elif family == "nanbeige":
        native = NanbeigeForCausalLM(NanbeigeConfig(**settings))
    save_file(
        {name: value.detach().clone() for name, value in native.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(json.dumps(native.config.to_dict()))
    config = TransformerConfig(
        num_layers=native.config.num_hidden_layers,
        hidden_size=8,
        num_attention_heads=2,
        ffn_hidden_size=16,
        params_dtype=torch.float32,
    )
    args = Namespace(
        hf_checkpoint=str(tmp_path),
        rlt_model_revision="tiny-cpu",
        recompute_granularity=None,
        custom_model_provider_path=f"vime_plugins.{family}.model.model_provider",
        num_experts=None,
    )
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    monkeypatch.setattr(megatron.training.arguments, "core_transformer_config_from_args", lambda args: config)
    monkeypatch.setattr(mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(common, "get_transformer_layer_offset", lambda config: 0)
    actor = get_model_provider_func(args, "actor")()
    critic = get_model_provider_func(args, "critic")()
    assert actor.role == "actor" and critic.role == "critic"
    assert "lm_head.weight" not in dict(critic.named_parameters())
    head = critic.output_layer.weight.detach().clone()
    with torch.no_grad():
        for name, value in critic.named_parameters():
            if name != "output_layer.weight":
                value.zero_()
    load_model_hf_weights(args, [critic], tmp_path, config, ouro_hf_tensor)
    assert torch.equal(head, critic.output_layer.weight)
    expected = native.state_dict()
    for name, value in critic.named_parameters():
        if name != "output_layer.weight":
            assert torch.equal(value, expected[name]), name


@pytest.mark.parametrize("recompute", [False, True])
def test_critic_head_and_core_gradients(recompute):
    torch.manual_seed(42)
    native = OuroForCausalLM(
        OuroConfig(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            eos_token_id=None,
        )
    )
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2, ffn_hidden_size=16)
    actor = OuroMegatronModel(config, native)
    critic = OuroMegatronModel(config, copy.deepcopy(native), role="critic", recompute=recompute)
    tokens = torch.tensor([[1, 2, 3]])
    assert actor(tokens).shape == (1, 3, 11)
    values = critic(tokens)
    assert values.shape == (1, 3, 1)
    assert "lm_head.weight" not in dict(critic.named_parameters())
    loss = (values[:, :-1].squeeze(-1) - torch.tensor([[0.0, 1.0]])).square().mean()
    loss.backward()
    for name, parameter in critic.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert "early_exit_gate" in name
            assert parameter.grad is None
