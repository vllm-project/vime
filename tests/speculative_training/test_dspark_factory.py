from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from vime.backends.speculative_training.factories import speculators_dspark


class _FakeDSparkModel(torch.nn.Module):
    def __init__(self, *, target_layer_ids=(2, 14, 29), block_size=7):
        super().__init__()
        self.target_layer_ids = list(target_layer_ids)
        self.block_size = block_size
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.verifier_lm_head = torch.nn.Linear(4, 8, bias=False)
        self.verifier_norm = torch.nn.LayerNorm(4)
        self.confidence_head = torch.nn.Linear(4, 1)


def _args():
    return Namespace(
        draft_model_path="/models/dspark",
        draft_feature_layer_ids=[2, 14, 29],
        draft_dspark_block_size=7,
        draft_target_embedding_path="/models/target",
        hf_checkpoint=None,
        hidden_size=4,
    )


def _checkpoint_config():
    return {
        "speculators_model_type": "dspark",
        "speculators_config": None,
        "aux_hidden_state_layer_ids": [2, 14, 29],
        "block_size": 7,
        "mask_token_id": 7,
        "markov_rank": 2,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 4,
            "vocab_size": 8,
        },
    }


def _install_fake_speculators(
    monkeypatch,
    *,
    model_layers=(2, 14, 29),
    block_size=7,
    loading_info=None,
    checkpoint_config=None,
):
    source_config = checkpoint_config if checkpoint_config is not None else _checkpoint_config()
    model = _FakeDSparkModel(target_layer_ids=model_layers, block_size=block_size)
    calls = {}

    class FakeConfig:
        model_fields = set(_checkpoint_config()) | {"architectures", "draft_vocab_size", "speculators_version"}

        @classmethod
        def from_dict(cls, value):
            calls["config_dict"] = value
            transformer = value["transformer_layer_config"]
            return SimpleNamespace(
                transformer_layer_config=SimpleNamespace(**transformer),
                aux_hidden_state_layer_ids=value.get("aux_hidden_state_layer_ids"),
                target_hidden_size=transformer["hidden_size"],
                markov_head_type=value.get("markov_head_type", "vanilla"),
                mask_token_id=value.get("mask_token_id"),
                markov_rank=value.get("markov_rank", 0),
                enable_confidence_head=value.get("enable_confidence_head", False),
                confidence_head_with_markov=value.get("confidence_head_with_markov", False),
                speculators_config=SimpleNamespace(verifier=SimpleNamespace(name_or_path="original")),
            )

    class FakeModelType:
        @classmethod
        def from_pretrained(cls, path, *, config, output_loading_info):
            calls["load"] = (path, config, output_loading_info)
            return model, (loading_info or {"missing_keys": [], "mismatched_keys": []})

    monkeypatch.setattr(speculators_dspark, "load_draft_checkpoint_config", lambda args: source_config)
    monkeypatch.setattr(speculators_dspark, "_load_speculators_types", lambda: (FakeConfig, FakeModelType))
    return model, source_config, calls


@pytest.mark.unit
def test_dspark_factory_restores_nested_config_from_vllm_schema(monkeypatch):
    checkpoint_config = _checkpoint_config()
    transformer_config = checkpoint_config["transformer_layer_config"]
    checkpoint_config.update(transformer_config)
    checkpoint_config["transformer_layer_config"] = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "vocab_size": transformer_config["vocab_size"],
    }
    checkpoint_config.pop("aux_hidden_state_layer_ids")
    checkpoint_config["target_layer_ids"] = [1, 13, 28]
    checkpoint_config["architectures"] = ["Qwen3DSparkModel"]
    model, _, calls = _install_fake_speculators(monkeypatch, checkpoint_config=checkpoint_config)

    result = speculators_dspark.build_model(_args(), torch.device("cpu"))

    assert result is model
    assert calls["config_dict"]["transformer_layer_config"] == transformer_config
    assert calls["config_dict"]["aux_hidden_state_layer_ids"] == [2, 14, 29]
    assert calls["config_dict"]["draft_vocab_size"] == transformer_config["vocab_size"]


@pytest.mark.unit
def test_dspark_factory_builds_canonical_model(monkeypatch):
    model, source_config, calls = _install_fake_speculators(
        monkeypatch,
        loading_info={
            "missing_keys": [
                "embed_tokens.weight",
                "lm_head.weight",
                "verifier_lm_head.weight",
                "verifier_norm.weight",
            ],
            "mismatched_keys": [],
        },
    )

    result = speculators_dspark.build_model(_args(), torch.device("cpu"))

    assert result is model
    assert calls["load"][0] == "/models/dspark"
    assert calls["load"][2] is True
    assert calls["config_dict"]["speculators_model_type"] == "dspark"
    assert "speculators_config" not in calls["config_dict"]
    assert source_config["speculators_config"] is None
    assert calls["load"][1].speculators_config.verifier.name_or_path == "/models/target"
    assert isinstance(model.verifier_norm, torch.nn.Identity)
    assert not model.lm_head.weight.requires_grad
    assert not model.verifier_lm_head.weight.requires_grad
    assert model.proj.weight.dtype == torch.bfloat16
    assert model.confidence_head.weight.dtype == torch.float32


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_layers", "block_size", "message"),
    [
        ((2, 10, 29), 7, "feature layers"),
        ((2, 14, 29), 8, "block size"),
    ],
)
def test_dspark_factory_rejects_loaded_model_layout_mismatch(
    monkeypatch,
    model_layers,
    block_size,
    message,
):
    _install_fake_speculators(
        monkeypatch,
        model_layers=model_layers,
        block_size=block_size,
    )

    with pytest.raises(ValueError, match=message):
        speculators_dspark.build_model(_args(), torch.device("cpu"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loading_info", "bad_key"),
    [
        (
            {"missing_keys": ["layers.0.self_attn.q_proj.weight"], "mismatched_keys": []},
            "layers.0.self_attn.q_proj.weight",
        ),
        (
            {
                "missing_keys": ["lm_head.weight"],
                "mismatched_keys": [("layers.0.mlp.up_proj.weight", (8, 4), (12, 4))],
            },
            "layers.0.mlp.up_proj.weight",
        ),
    ],
)
def test_dspark_factory_rejects_non_target_loading_errors(monkeypatch, loading_info, bad_key):
    _install_fake_speculators(monkeypatch, loading_info=loading_info)

    with pytest.raises(ValueError, match="incomplete or incompatible") as error:
        speculators_dspark.build_model(_args(), torch.device("cpu"))

    assert bad_key in str(error.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
