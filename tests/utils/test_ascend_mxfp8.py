"""CPU unit tests for Ascend W8A8-MXFP8 rollout weight reload helpers."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from vime.backends.vllm_utils import ascend_mxfp8 as mx


class _FakeAscendModelSlimConfig:
    def __init__(self, quant_description):
        self.quant_description = quant_description


class _FloatConfig:
    quant_description = {"quant_method": "ascend", "layer.weight": "FLOAT"}


class _FakeScheme:
    def __init__(self):
        self.restore_weights_for_rl_loading = MagicMock()
        self.process_weights_after_loading = MagicMock()


class _QuantWrapper:
    def __init__(self, scheme):
        self.quant_method = scheme


class _Linear(torch.nn.Module):
    def __init__(self, *, transformed: bool = False, native_shapes_restored: bool = True):
        super().__init__()
        self.scheme = _FakeScheme()
        self.quant_method = _QuantWrapper(self.scheme)
        if transformed and not native_shapes_restored:
            weight_shape = (4, 2)
            scale_shape = (2, 2, 2)
        else:
            weight_shape = (2, 4)
            scale_shape = (2, 4)
        self.weight = torch.nn.Parameter(torch.zeros(weight_shape), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.zeros(scale_shape), requires_grad=False)
        self._mxfp8_original_shapes = {"weight": (2, 4), "weight_scale": (2, 4)}
        self._mxfp8_transformed = transformed


class _FloatLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)


class _Block(torch.nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.proj = proj


class _Model(torch.nn.Module):
    def __init__(self, proj):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(proj)])
        self.packed_modules_mapping = {}


class _PackedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = _Linear()


class _PackedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_PackedBlock()])
        self.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


class _FusedMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scheme = _FakeScheme()
        self.quant_method = _QuantWrapper(self.scheme)
        self.w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 4), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.zeros(2, 4, 4), requires_grad=False)


class _MoEBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _FusedMoE()


class _MoEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_MoEBlock()])
        self.packed_modules_mapping = {}


@pytest.fixture
def fake_ascend_config_module(monkeypatch):
    package = types.ModuleType("vllm_ascend")
    package.__path__ = []
    quantization = types.ModuleType("vllm_ascend.quantization")
    quantization.__path__ = []
    modelslim = types.ModuleType("vllm_ascend.quantization.modelslim_config")
    modelslim.AscendModelSlimConfig = _FakeAscendModelSlimConfig
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quantization)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.modelslim_config", modelslim)
    return _FakeAscendModelSlimConfig


@pytest.fixture
def fake_torch_npu(monkeypatch):
    module = types.ModuleType("torch_npu")
    module.float8_e4m3fn = torch.float8_e4m3fn
    module.npu_dynamic_mx_quant = MagicMock(
        return_value=(torch.zeros(2, 4, dtype=torch.float8_e4m3fn), torch.arange(6, dtype=torch.uint8).view(2, 3, 1))
    )
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return module


@pytest.mark.unit
def test_detects_only_modelslim_config_with_mxfp8(fake_ascend_config_module):
    assert mx.is_ascend_mxfp8_config(
        fake_ascend_config_module({"quant_method": "ascend", "layers.0.proj.weight": "W8A8_MXFP8"})
    )
    assert not mx.is_ascend_mxfp8_config(fake_ascend_config_module({"quant_method": "ascend", "x": "FLOAT"}))
    assert not mx.is_ascend_mxfp8_config(_FloatConfig())


@pytest.mark.unit
def test_quantize_mxfp8_linear_weight_emits_weight_and_scale(fake_torch_npu):
    model = _Model(_Linear())
    source = torch.ones(2, 4)

    output = list(mx.quantize_mxfp8_weights([("layers.0.proj.weight", source)], model, torch.bfloat16))

    assert [name for name, _ in output] == ["layers.0.proj.weight", "layers.0.proj.weight_scale"]
    fake_torch_npu.npu_dynamic_mx_quant.assert_called_once()
    call = fake_torch_npu.npu_dynamic_mx_quant.call_args
    assert torch.equal(call.args[0], source.to(torch.bfloat16))
    assert call.kwargs == {"axis": -1, "dst_type": torch.float8_e4m3fn}
    assert output[1][1].shape == (2, 3)


@pytest.mark.unit
def test_non_mxfp8_weight_passes_through_unchanged(fake_torch_npu):
    model = _Model(_FloatLinear())
    source = torch.ones(2, 4)

    output = list(mx.quantize_mxfp8_weights([("layers.0.proj.weight", source)], model, torch.bfloat16))

    assert len(output) == 1
    assert output[0][0] == "layers.0.proj.weight"
    assert output[0][1] is source
    fake_torch_npu.npu_dynamic_mx_quant.assert_not_called()


@pytest.mark.unit
def test_packed_linear_source_name_maps_to_quantized_target(fake_torch_npu):
    output = list(
        mx.quantize_mxfp8_weights(
            [("layers.0.q_proj.weight", torch.ones(2, 4))],
            _PackedModel(),
        )
    )

    assert [name for name, _ in output] == ["layers.0.q_proj.weight", "layers.0.q_proj.weight_scale"]


@pytest.mark.unit
def test_fused_moe_source_path_maps_to_quantized_target(fake_torch_npu):
    output = list(
        mx.quantize_mxfp8_weights(
            [("layers.0.mlp.experts.0.gate_proj.weight", torch.ones(2, 4))],
            _MoEModel(),
        )
    )

    assert [name for name, _ in output] == [
        "layers.0.mlp.experts.0.gate_proj.weight",
        "layers.0.mlp.experts.0.gate_proj.weight_scale",
    ]


@pytest.mark.unit
def test_prepare_uses_scheme_restore_when_native_metadata_has_not_restored_shapes():
    layer = _Linear(transformed=True, native_shapes_restored=False)
    layer.scheme.restore_weights_for_rl_loading.side_effect = lambda module: setattr(
        module, "_mxfp8_transformed", False
    )
    model = _Model(layer)

    assert mx.prepare_mxfp8_modules_for_reload(model) == 1
    layer.scheme.restore_weights_for_rl_loading.assert_called_once_with(layer)
    assert layer._mxfp8_transformed is False


@pytest.mark.unit
def test_prepare_only_resets_marker_when_native_reload_already_restored_shapes():
    layer = _Linear(transformed=True, native_shapes_restored=True)
    model = _Model(layer)

    assert mx.prepare_mxfp8_modules_for_reload(model) == 1
    layer.scheme.restore_weights_for_rl_loading.assert_not_called()
    assert layer._mxfp8_transformed is False


@pytest.mark.unit
def test_finalize_reapplies_mxfp8_layout_idempotently():
    layer = _Linear(transformed=False)
    layer.scheme.process_weights_after_loading.side_effect = lambda module: setattr(
        module, "_mxfp8_transformed", True
    )
    model = _Model(layer)

    assert mx.finalize_mxfp8_modules_after_reload(model) == 1
    assert mx.finalize_mxfp8_modules_after_reload(model) == 0
    layer.scheme.process_weights_after_loading.assert_called_once_with(layer)
