from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from vime.backends.speculative_training.factories import speculators_eagle3


class _FakeEagle3Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(12, 4, bias=False)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 4, bias=False)])
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 6, bias=False)
        self.verifier_lm_head = torch.nn.Linear(4, 6, bias=False)
        self.verifier_norm = torch.nn.LayerNorm(4)
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.missing_keys: list[str] = []
        self.unexpected_keys: list[str] = []
        self.calls: dict = {}

    def load_state_dict(self, state_dict, strict=True):
        self.calls["state_dict"] = dict(state_dict)
        return SimpleNamespace(
            missing_keys=list(self.missing_keys),
            unexpected_keys=list(self.unexpected_keys),
        )

    def load_vocab_mappings(self, t2d, d2t):
        self.calls["vocab"] = (t2d, d2t)

    def load_verifier_weights(self):
        self.calls["verifier"] = True

    def to(self, *args, **kwargs):
        self.calls["to"] = (args, kwargs)
        return self


def _args(tmp_path) -> Namespace:
    return Namespace(
        draft_model_path=str(tmp_path),
        draft_feature_layer_ids=[1, 2, 3],
        draft_target_embedding_path="/models/target",
        hf_checkpoint=None,
        hidden_size=32,
        draft_ttt_length=1,
    )


def _checkpoint_config() -> dict:
    return {
        "model_type": "llama",
        "hidden_size": 32,
        "vocab_size": 8,
        "draft_vocab_size": 6,
        "num_hidden_layers": 1,
        "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
    }


def _checkpoint_weights() -> dict:
    return {
        "midlayer.self_attn.q_proj.weight": torch.zeros(4, 8),
        "midlayer.mlp.gate_proj.weight": torch.zeros(4, 4),
        "fc.weight": torch.zeros(4, 12),
        "norm.weight": torch.ones(4),
        "lm_head.weight": torch.zeros(6, 4),
        "t2d": torch.tensor([True] * 6 + [False] * 2),
        "d2t": torch.arange(6),
    }


def _install_fake_speculators(
    monkeypatch,
    tmp_path,
    *,
    checkpoint_config=None,
    checkpoint_weights=None,
):
    model = _FakeEagle3Model()
    source_config = checkpoint_config if checkpoint_config is not None else _checkpoint_config()
    weights = checkpoint_weights if checkpoint_weights is not None else _checkpoint_weights()
    (tmp_path / "model.safetensors").write_bytes(b"")
    calls = model.calls

    class FakeEagle3SpeculatorConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSpeculatorsConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls["speculators_config"] = kwargs

    class FakeGreedyProposalConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeVerifierConfig:
        @classmethod
        def from_pretrained(cls, path):
            calls["verifier_path"] = path
            return SimpleNamespace(name_or_path=path)

    class FakeModelType:
        def __new__(cls, config):
            model.config = config
            calls["model_config"] = config
            return model

    monkeypatch.setattr(speculators_eagle3, "load_draft_checkpoint_config", lambda args: source_config)
    monkeypatch.setattr(speculators_eagle3, "load_file", lambda path, device="cpu": dict(weights))
    monkeypatch.setattr(
        speculators_eagle3,
        "_load_speculators_types",
        lambda: (
            FakeEagle3SpeculatorConfig,
            FakeModelType,
            FakeSpeculatorsConfig,
            FakeGreedyProposalConfig,
            FakeVerifierConfig,
        ),
    )
    return model, calls


@pytest.mark.unit
def test_eagle3_factory_builds_canonical_speculators_model(monkeypatch, tmp_path):
    model, calls = _install_fake_speculators(monkeypatch, tmp_path)

    result = speculators_eagle3.build_model(_args(tmp_path), torch.device("cpu"))

    assert result is model
    assert calls["verifier_path"] == "/models/target"
    assert calls["speculators_config"]["algorithm"] == "eagle3"
    assert calls["speculators_config"]["default_proposal_method"] == "greedy"
    (proposal,) = calls["speculators_config"]["proposal_methods"]
    assert proposal.speculative_tokens == 1
    # The vLLM midlayer layout is remapped and the vocab buffers bypass
    # load_state_dict through their dedicated API.
    assert set(calls["state_dict"]) == {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "fc.weight",
        "norm.weight",
        "lm_head.weight",
    }
    t2d, d2t = calls["vocab"]
    assert torch.equal(t2d, _checkpoint_weights()["t2d"].to(torch.bool))
    assert torch.equal(d2t, _checkpoint_weights()["d2t"].to(torch.long))
    assert calls["verifier"] is True
    assert isinstance(model.verifier_norm, torch.nn.Identity)
    assert model.lm_head.weight.requires_grad
    assert calls["to"][1] == {"device": torch.device("cpu"), "dtype": torch.bfloat16}
    # CPU keeps the model's own attention implementation choice.
    assert calls["model_config"].transformer_layer_config._attn_implementation is None


class _FakeNpuDevice:
    """torch.device("npu") needs the torch_npu backend, which CPU-only CI
    lacks; the factory only reads .type and the fake speculators model only
    records the object, so a plain stub exercises the NPU branch anywhere.
    """

    type = "npu"


@pytest.mark.unit
def test_eagle3_factory_pins_eager_attention_on_npu(monkeypatch, tmp_path):
    _, calls = _install_fake_speculators(monkeypatch, tmp_path)
    device = _FakeNpuDevice()

    speculators_eagle3.build_model(_args(tmp_path), device)

    assert calls["model_config"].transformer_layer_config._attn_implementation == "eager"
    assert calls["to"][1] == {"device": device, "dtype": torch.bfloat16}


@pytest.mark.unit
def test_eagle3_factory_skips_vocab_api_for_full_vocab_checkpoints(monkeypatch, tmp_path):
    weights = _checkpoint_weights()
    weights.pop("t2d")
    weights.pop("d2t")
    model, calls = _install_fake_speculators(monkeypatch, tmp_path, checkpoint_weights=weights)

    speculators_eagle3.build_model(_args(tmp_path), torch.device("cpu"))

    assert "vocab" not in calls
    assert calls["verifier"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("checkpoint_config", "message"),
    [
        ({"model_type": "qwen3", "hidden_size": 32}, "Llama-family"),
        ({**_checkpoint_config(), "draft_vocab_size": None}, "draft_vocab_size"),
        (
            {**_checkpoint_config(), "eagle_aux_hidden_state_layer_ids": [1, 2, 4]},
            "eagle_aux_hidden_state_layer_ids",
        ),
        (
            {k: v for k, v in _checkpoint_config().items() if k != "eagle_aux_hidden_state_layer_ids"},
            "eagle_aux_hidden_state_layer_ids",
        ),
    ],
)
def test_eagle3_factory_rejects_incompatible_checkpoint_configs(monkeypatch, tmp_path, checkpoint_config, message):
    _install_fake_speculators(monkeypatch, tmp_path, checkpoint_config=checkpoint_config)

    with pytest.raises(ValueError, match=message):
        speculators_eagle3.build_model(_args(tmp_path), torch.device("cpu"))


@pytest.mark.unit
def test_eagle3_factory_rejects_hidden_size_mismatch(monkeypatch, tmp_path):
    _install_fake_speculators(monkeypatch, tmp_path)
    args = _args(tmp_path)
    args.hidden_size = 8

    with pytest.raises(ValueError, match="hidden sizes"):
        speculators_eagle3.build_model(args, torch.device("cpu"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys", "bad_key"),
    [
        (["fc.weight"], [], "fc.weight"),
        ([], ["layers.0.hidden_norm.weight"], "layers.0.hidden_norm.weight"),
    ],
)
def test_eagle3_factory_rejects_checkpoint_layout_mismatch(
    monkeypatch, tmp_path, missing_keys, unexpected_keys, bad_key
):
    model, _ = _install_fake_speculators(monkeypatch, tmp_path)
    model.missing_keys = missing_keys
    model.unexpected_keys = unexpected_keys

    with pytest.raises(RuntimeError, match="does not match") as error:
        speculators_eagle3.build_model(_args(tmp_path), torch.device("cpu"))

    assert bad_key in str(error.value)


@pytest.mark.unit
def test_eagle3_factory_requires_model_safetensors(monkeypatch, tmp_path):
    _install_fake_speculators(monkeypatch, tmp_path)
    args = _args(tmp_path / "elsewhere")

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        speculators_eagle3.build_model(args, torch.device("cpu"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
