from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def load_common_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.__path__ = []

    core_mod = types.ModuleType("megatron.core")
    core_mod.__path__ = []
    core_mod.mpu = types.SimpleNamespace(
        get_expert_tensor_parallel_world_size=lambda: 2,
        get_expert_tensor_parallel_group=lambda: "expert_tp",
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_group=lambda: "tp",
        get_expert_model_parallel_world_size=lambda: 1,
        get_expert_model_parallel_rank=lambda: 0,
    )

    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_mod.__path__ = []
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0
    transformer_mod.transformer_layer = transformer_layer_mod
    core_mod.transformer = transformer_mod
    megatron_mod.core = core_mod

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_layer", transformer_layer_mod)

    module_path = (
        Path(__file__).resolve().parents[1]
        / "slime"
        / "backends"
        / "megatron_utils"
        / "update_weight"
        / "common.py"
    )
    module_name = "test_update_weight_common_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_maybe_get_cpu_backup_returns_original_tensor_when_hook_missing(monkeypatch):
    common = load_common_module(monkeypatch)

    monkeypatch.setitem(sys.modules, "torch_memory_saver", types.ModuleType("torch_memory_saver"))

    tensor = torch.tensor([1, 2, 3])
    assert common._maybe_get_cpu_backup(tensor) is tensor
