import argparse
import sys
import types
from argparse import Namespace

import pytest

from vime.backends.megatron_utils.update_weight import create_weight_updater

from vime.utils.arguments import get_vime_extra_args_provider

NUM_GPUS = 0


class _FakeUpdater:
    def __init__(self, args, model, weights_getter, *, model_name, quantization_config):
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "transport", "colocate", "module_name", "class_name"),
    [
        pytest.param("delta", "disk", False, "update_weight_from_disk_delta", "UpdateWeightFromDiskDelta", id="delta"),
        pytest.param(
            "delta",
            "sparse_hccl",
            False,
            "update_weight_from_sparse_hccl",
            "UpdateWeightFromSparseHCCL",
            id="sparse-hccl-delta",
        ),
        pytest.param("full", "disk", False, "update_weight_from_disk", "UpdateWeightFromDisk", id="disk"),
        pytest.param("full", "nccl", True, "update_weight_from_tensor", "UpdateWeightFromTensor", id="colocated"),
        pytest.param(
            "full",
            "nccl",
            False,
            "update_weight_from_distributed",
            "UpdateWeightFromDistributed",
            id="distributed",
        ),
    ],
)
def test_create_weight_updater_selects_implementation(monkeypatch, mode, transport, colocate, module_name, class_name):
    full_module_name = f"vime.backends.megatron_utils.update_weight.{module_name}"
    fake_module = types.ModuleType(full_module_name)
    setattr(fake_module, class_name, _FakeUpdater)
    monkeypatch.setitem(sys.modules, full_module_name, fake_module)

    args = Namespace(
        update_weight_mode=mode,
        update_weight_transport=transport,
        update_weight_start_version=7,
        colocate=colocate,
    )
    model = [object()]

    def weights_getter():
        return {"weight": object()}

    updater = create_weight_updater(
        args,
        model,
        weights_getter,
        model_name="model",
        quantization_config={"quant_method": "test"},
    )

    assert isinstance(updater, _FakeUpdater)
    assert updater.args is args
    assert updater.model is model
    assert updater.weights_getter is weights_getter
    assert updater.model_name == "model"
    assert updater.quantization_config == {"quant_method": "test"}
    assert updater.weight_version == 7


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["nccl", "disk", "sparse_hccl"])
def test_transport_options_are_registered_once(transport):
    parser = get_vime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--update-weight-transport",
            transport,
            "--update-weight-delta-verify-every",
            "1",
        ]
    )
    assert args.update_weight_transport == transport
    assert args.update_weight_delta_verify_every == 1
    assert args.update_weight_delta_batch_diff == 32
    assert args.update_weight_delta_batch_gather == 32
