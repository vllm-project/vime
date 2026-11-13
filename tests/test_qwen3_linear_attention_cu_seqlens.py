from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GPUS = 0


def install_megatron_stubs() -> None:
    if "megatron" in sys.modules:
        return

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    inference_mod = types.ModuleType("megatron.core.inference")
    inference_contexts_mod = types.ModuleType("megatron.core.inference.contexts")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")
    parallel_state_mod = types.ModuleType("megatron.core.parallel_state")
    tensor_parallel_layers_mod = types.ModuleType("megatron.core.tensor_parallel.layers")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_module_mod = types.ModuleType("megatron.core.transformer.module")
    spec_utils_mod = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_block_mod = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    utils_mod = types.ModuleType("megatron.core.utils")

    class PackedSeqParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class ModuleSpec:
        def __init__(self, module=None, params=None):
            self.module = module
            self.params = params or {}

    class FakeParallelLinear(nn.Module):
        def __init__(self, in_features, out_features, config=None, init_method=None, bias=True, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))

        def forward(self, x):
            return F.linear(x, self.weight), None

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    tensor_parallel_stub = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )

    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    inference_contexts_mod.BaseInferenceContext = type("BaseInferenceContext", (), {})
    packed_seq_mod.PackedSeqParams = PackedSeqParams
    parallel_state_mod.get_tensor_model_parallel_world_size = lambda: 1
    tensor_parallel_layers_mod.ColumnParallelLinear = FakeParallelLinear
    tensor_parallel_layers_mod.RowParallelLinear = FakeParallelLinear
    transformer_module_mod.MegatronModule = MegatronModule
    spec_utils_mod.ModuleSpec = ModuleSpec
    transformer_block_mod.get_num_layers_to_build = lambda *args, **kwargs: 0
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0
    utils_mod.divide = lambda x, y: x // y
    utils_mod.init_method_normal = lambda std: (lambda tensor: tensor)

    core_mod.mpu = mpu_stub
    core_mod.tensor_parallel = tensor_parallel_stub

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["megatron.core.inference"] = inference_mod
    sys.modules["megatron.core.inference.contexts"] = inference_contexts_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod
    sys.modules["megatron.core.parallel_state"] = parallel_state_mod
    sys.modules["megatron.core.tensor_parallel.layers"] = tensor_parallel_layers_mod
    sys.modules["megatron.core.transformer"] = transformer_mod
    sys.modules["megatron.core.transformer.module"] = transformer_module_mod
    sys.modules["megatron.core.transformer.spec_utils"] = spec_utils_mod
    sys.modules["megatron.core.transformer.transformer_block"] = transformer_block_mod
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_mod
    sys.modules["megatron.core.utils"] = utils_mod


class FakeShortConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, cu_seqlens=None, **kwargs):
        return x, None


class FakeFusedRMSNormGated(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, z):
        return x


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.float32,
    )


def load_module(module_name: str):
    install_megatron_stubs()
    sys.modules.pop("vime_plugins.models.hf_attention", None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "class_name", "args", "expected_backend"),
    [
        ("vime_plugins.models.qwen3_5", "Qwen3_5GatedDeltaNet", None, "fla_npu"),
        (
            "vime_plugins.models.qwen3_5",
            "Qwen3_5GatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
        ("vime_plugins.models.qwen3_next", "Qwen3NextGatedDeltaNet", None, "fla"),
        (
            "vime_plugins.models.qwen3_next",
            "Qwen3NextGatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
    ],
)
def test_linear_attention_forwards_cu_seqlens_to_chunk_kernel(
    monkeypatch,
    module_name: str,
    class_name: str,
    args,
    expected_backend: str,
):
    module = load_module(module_name)

    monkeypatch.setattr(module.accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(module, "ShortConvolution", FakeShortConvolution, raising=False)
    monkeypatch.setattr(module, "FusedRMSNormGated", FakeFusedRMSNormGated, raising=False)
    monkeypatch.setattr(module, "_FlaNpuShortConvolution", FakeShortConvolution, raising=False)
    monkeypatch.setattr(module, "_FlaNpuRMSNormGated", FakeFusedRMSNormGated, raising=False)

    chunk_calls = []
    selected_backends = []

    def fake_chunk_gated_delta_rule(
        q,
        k,
        v,
        *,
        g,
        beta,
        use_qk_l2norm_in_kernel,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        **kwargs,
    ):
        chunk_calls.append(cu_seqlens.clone() if cu_seqlens is not None else None)
        assert q.shape[0] == 1
        assert cu_seqlens is not None
        return torch.zeros_like(v), None

    def fake_get_chunk_gated_delta_rule(backend):
        selected_backends.append(backend)
        return fake_chunk_gated_delta_rule

    monkeypatch.setattr(module, "get_chunk_gated_delta_rule", fake_get_chunk_gated_delta_rule)

    layer = getattr(module, class_name)(make_config(), layer_idx=0, args=args)
    hidden_states = torch.randn(1, 7, 32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    output = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert selected_backends == [expected_backend]
    assert layer.gdn_backend == expected_backend
    assert output.shape == hidden_states.shape
    assert len(chunk_calls) == 1
    assert torch.equal(chunk_calls[0], cu_seqlens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
