"""Unit tests for the GLM-4.7-Flash (glm4_moe_lite) megatron bridge mapping registry.

The real ``glm4moe_lite.py`` imports the full ``megatron.bridge`` + ``transformers``
stack, which is unavailable on bare-deps dev machines. We install lightweight
stubs for those modules (mirroring the approach in ``test_qwen3_5_mtp_bridge_mapping.py``),
then drive ``GLM47MTPBridge.mapping_registry`` directly. This locks in the
MLA-only behaviour introduced by removing the always-dead non-MLA branches:
the mapping table must always emit the individual MLA Q/KV down/up projections
and ``q_a_layernorm`` / ``kv_a_layernorm`` and must never emit a fused QKV.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Stub mapping classes — mirror the subset of the param_mapping API that
# ``mapping_registry`` actually constructs/reads back.
# ---------------------------------------------------------------------------


@dataclass
class AutoMapping:
    megatron_param: str
    hf_param: str | None = None

    @classmethod
    def register_module_type(cls, _name: str, _parallelism_type: str) -> None:
        # No-op: the real class registers MindSpeed TE types for NPU weight
        # mapping; the registry itself is exercised at import time via
        # ``_register_mindspeed_te_module_types`` and must not crash.
        return


@dataclass
class GatedMLPMapping:
    megatron_param: str
    gate: str | None = None
    up: str | None = None


@dataclass
class GLMExpertGateUpProjMapping:
    megatron_param: str
    hf_param: str | None = None


@dataclass
class GLMExpertDownProjMapping:
    megatron_param: str
    hf_param: str | None = None


class MegatronMappingRegistry:
    """Minimal container exposing iteration + membership over its mappings."""

    def __init__(self, *mappings: Any) -> None:
        self.mappings = list(mappings)

    def __iter__(self):
        return iter(self.mappings)

    def __len__(self) -> int:
        return len(self.mappings)


def _install_stubs() -> None:
    """Install stub modules for megatron.bridge / transformers / transformer_engine."""
    # --- megatron.core.models.gpt ---
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_mod.GPTModel = type("GPTModel", (), {})

    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: "decoder_block_spec"

    core_mod = types.ModuleType("megatron.core")
    core_mod.__path__ = []  # type: ignore[attr-defined]
    core_mod.models = types.ModuleType("megatron.core.models")
    core_mod.models.gpt = gpt_mod  # type: ignore[attr-defined]
    core_mod.models.gpt.gpt_layer_specs = gpt_layer_specs_mod  # type: ignore[attr-defined]

    megatron_mod = types.ModuleType("megatron")
    megatron_mod.__path__ = []  # type: ignore[attr-defined]
    megatron_mod.core = core_mod

    # --- megatron.bridge.* (conversion + glm + providers) ---
    param_mapping_mod = types.ModuleType("megatron.bridge.models.conversion.param_mapping")
    param_mapping_mod.AutoMapping = AutoMapping
    param_mapping_mod.GatedMLPMapping = GatedMLPMapping

    mapping_registry_mod = types.ModuleType("megatron.bridge.models.conversion.mapping_registry")
    mapping_registry_mod.MegatronMappingRegistry = MegatronMappingRegistry

    model_bridge_mod = types.ModuleType("megatron.bridge.models.conversion.model_bridge")

    def _register_bridge(source=None, target=None, model_type=None):
        def decorator(cls):
            return cls

        return decorator

    model_bridge_mod.MegatronModelBridge = type(
        "MegatronModelBridge", (), {"register_bridge": staticmethod(_register_bridge)}
    )

    glm_moe_mappings_mod = types.ModuleType("megatron.bridge.models.glm.glm_moe_mappings")
    glm_moe_mappings_mod.GLMExpertGateUpProjMapping = GLMExpertGateUpProjMapping
    glm_moe_mappings_mod.GLMExpertDownProjMapping = GLMExpertDownProjMapping

    glm45_bridge_mod = types.ModuleType("megatron.bridge.models.glm.glm45_bridge")

    class GLM45Bridge:
        # The real base carries the fused-expert / suffix helpers; we only need
        # the bridge subclass to *inherit* the name. ``mapping_registry`` is fully
        # defined on the subclass, so the base body never runs here.
        pass

    glm45_bridge_mod.GLM45Bridge = GLM45Bridge

    causal_lm_mod = types.ModuleType("megatron.bridge.models.hf_pretrained.causal_lm")
    causal_lm_mod.PreTrainedCausalLM = type("PreTrainedCausalLM", (), {})

    mla_provider_mod = types.ModuleType("megatron.bridge.models.mla_provider")
    mla_provider_mod.MLAModelProvider = type("MLAModelProvider", (), {})

    gpt_provider_mod = types.ModuleType("megatron.bridge.models.gpt_provider")
    gpt_provider_mod.GPTModelProvider = type("GPTModelProvider", (), {})

    for name, mod in [
        ("megatron", megatron_mod),
        ("megatron.core", core_mod),
        ("megatron.core.models", core_mod.models),
        ("megatron.core.models.gpt", gpt_mod),
        ("megatron.core.models.gpt.gpt_layer_specs", gpt_layer_specs_mod),
        ("megatron.bridge", types.ModuleType("megatron.bridge")),
        ("megatron.bridge.models", types.ModuleType("megatron.bridge.models")),
        ("megatron.bridge.models.conversion", types.ModuleType("megatron.bridge.models.conversion")),
        ("megatron.bridge.models.conversion.param_mapping", param_mapping_mod),
        ("megatron.bridge.models.conversion.mapping_registry", mapping_registry_mod),
        ("megatron.bridge.models.conversion.model_bridge", model_bridge_mod),
        ("megatron.bridge.models.glm", types.ModuleType("megatron.bridge.models.glm")),
        ("megatron.bridge.models.glm.glm_moe_mappings", glm_moe_mappings_mod),
        ("megatron.bridge.models.glm.glm45_bridge", glm45_bridge_mod),
        ("megatron.bridge.models.hf_pretrained", types.ModuleType("megatron.bridge.models.hf_pretrained")),
        ("megatron.bridge.models.hf_pretrained.causal_lm", causal_lm_mod),
        ("megatron.bridge.models.mla_provider", mla_provider_mod),
        ("megatron.bridge.models.gpt_provider", gpt_provider_mod),
    ]:
        sys.modules.setdefault(name, mod)

    # --- transformers.Glm4MoeLiteForCausalLM (registration target) ---
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.Glm4MoeLiteForCausalLM = type("Glm4MoeLiteForCausalLM", (), {})
    sys.modules.setdefault("transformers", transformers_mod)

    # --- transformer_engine: absent on dev machine; import is try/except-guarded ---
    sys.modules.pop("transformer_engine", None)


_MODULE_NAME = "test_glm4_7_mtp_bridge_mapping_module"


def load_bridge_module():
    _install_stubs()
    module_path = Path(__file__).resolve().parents[1] / "vime_plugins" / "megatron_bridge" / "glm4moe_lite.py"
    sys.modules.pop(_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class _FakeHFConfig:
    """Minimal GLM-4.7-Flash HF config covering the fields mapping_registry reads."""

    num_hidden_layers: int = 47
    num_nextn_predict_layers: int = 1
    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 1536
    n_shared_experts: int = 1
    q_lora_rank: int = 768


def _make_bridge(module, *, hf_config=None, fused_experts=False):
    """Instantiate GLM47MTPBridge without running the heavy base __init__."""
    bridge = object.__new__(module.GLM47MTPBridge)
    if hf_config is not None:
        bridge._hf_config = hf_config
    # Override the inherited helpers so we never touch the real GLM45Bridge.
    bridge._uses_fused_experts = lambda: fused_experts
    bridge._hf_expert_suffix = lambda name: ""
    return bridge


def _megatron_params(registry) -> set[str]:
    return {m.megatron_param for m in registry}


def _hf_params(registry) -> set[str]:
    out = set()
    for m in registry:
        if getattr(m, "hf_param", None) is not None:
            out.add(m.hf_param)
        if getattr(m, "gate", None) is not None:
            out.add(m.gate)
        if getattr(m, "up", None) is not None:
            out.add(m.up)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_returns_nonempty_mapping_table():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig())

    registry = bridge.mapping_registry()

    assert isinstance(registry, MegatronMappingRegistry)
    assert len(registry) > 0


@pytest.mark.unit
def test_dense_layer_mappings_use_mla_q_a_layernorm_not_q_norm():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig())

    registry = bridge.mapping_registry()
    hf = _hf_params(registry)

    # MLA simplification (review note 4): q_layernorm maps straight to q_a_layernorm.
    assert "model.layers.*.self_attn.q_a_layernorm.weight" in hf
    assert "model.layers.*.self_attn.q_norm.weight" not in hf


@pytest.mark.unit
def test_registry_emits_individual_mla_projections_and_never_fused_qkv():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig())

    registry = bridge.mapping_registry()
    hf = _hf_params(registry)

    # MLA (review note 3/5): individual Q/KV down/up projections, not a fused QKV.
    for hf_name in (
        "model.layers.*.self_attn.q_a_proj.weight",
        "model.layers.*.self_attn.q_b_proj.weight",
        "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.*.self_attn.kv_b_proj.weight",
        "model.layers.*.self_attn.q_a_layernorm.weight",
        "model.layers.*.self_attn.kv_a_layernorm.weight",
    ):
        assert hf_name in hf, f"missing MLA projection mapping: {hf_name}"

    # The removed non-MLA branch would have produced a fused q_proj/k_proj/v_proj.
    for fused in (
        "model.layers.*.self_attn.q_proj.weight",
        "model.layers.*.self_attn.k_proj.weight",
        "model.layers.*.self_attn.v_proj.weight",
    ):
        assert fused not in hf, f"fused QKV mapping should not exist for MLA model: {fused}"


@pytest.mark.unit
def test_mtp_mappings_present_when_config_has_nextn_layers():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig(num_hidden_layers=47, num_nextn_predict_layers=1))

    registry = bridge.mapping_registry()
    megatron = _megatron_params(registry)
    hf = _hf_params(registry)

    # num_hidden_layers=47, mtp layer 0 -> HF layer index 47.
    for mtp_megatron in (
        "mtp.layers.0.enorm.weight",
        "mtp.layers.0.hnorm.weight",
        "mtp.layers.0.eh_proj.weight",
        "mtp.layers.0.final_layernorm.weight",
    ):
        assert mtp_megatron in megatron, f"missing MTP mapping: {mtp_megatron}"

    # MTP transformer layer reuses the MLA spec -> individual projections at index 47.
    for mtp_hf in (
        "model.layers.47.self_attn.q_a_proj.weight",
        "model.layers.47.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.47.self_attn.q_a_layernorm.weight",
    ):
        assert mtp_hf in hf, f"missing MTP MLA projection: {mtp_hf}"


@pytest.mark.unit
def test_mtp_mappings_absent_when_no_nextn_layers():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig(num_nextn_predict_layers=0))

    registry = bridge.mapping_registry()
    megatron = _megatron_params(registry)

    assert not any(p.startswith("mtp.") for p in megatron)


@pytest.mark.unit
def test_mtp_mappings_absent_when_hf_config_missing():
    # No self._hf_config -> early return after the dense/expert mappings.
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=None)

    registry = bridge.mapping_registry()
    megatron = _megatron_params(registry)

    assert not any(p.startswith("mtp.") for p in megatron)
    # Dense-layer MLA mappings are still produced.
    assert any("linear_q_down_proj" in p for p in megatron)


@pytest.mark.unit
def test_fused_experts_branch_uses_glm_expert_grouped_mappings():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig(), fused_experts=True)

    registry = bridge.mapping_registry()

    fused_types = {type(m).__name__ for m in registry}
    assert "GLMExpertGateUpProjMapping" in fused_types
    assert "GLMExpertDownProjMapping" in fused_types


@pytest.mark.unit
def test_non_fused_experts_branch_uses_gated_mlp_for_experts():
    module = load_bridge_module()
    bridge = _make_bridge(module, hf_config=_FakeHFConfig(), fused_experts=False)

    registry = bridge.mapping_registry()

    # Non-fused path emits a GatedMLPMapping for experts.linear_fc1 (with gate/up).
    expert_fc1 = [
        m
        for m in registry
        if getattr(m, "megatron_param", "").endswith("experts.linear_fc1.weight*")
        and type(m).__name__ == "GatedMLPMapping"
    ]
    assert expert_fc1, "non-fused experts should use GatedMLPMapping for fc1"
    assert expert_fc1[0].gate.endswith("experts.*.gate_proj.weight")
    assert expert_fc1[0].up.endswith("experts.*.up_proj.weight")
