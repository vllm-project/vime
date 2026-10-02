"""Exercise the Docker-patched native DistributedOptimizer methods on CUDA.

Run with torch.distributed.run and the patched Megatron tree on PYTHONPATH.
The small buffer fixture isolates common-state restoration; full checkpoint
and fresh-process coverage belongs to the Ouro/Qwen training recipe.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from megatron.core.optimizer import distrib_optimizer
from megatron.core.optimizer.optimizer import param_group_identifier_keys
from megatron.core.optimizer.optimizer_config import OptimizerConfig

NUM_GPUS = 1


def make_optimizer(kind):
    assert not distrib_optimizer.HAVE_APEX_OR_TE, "This test must exercise native Adam"
    parameters = [torch.nn.Parameter(torch.ones(size, device="cuda")) for size in (2, 3)]
    groups = [{"params": parameters}, {"params": []}]
    for index, group in enumerate(groups):
        group.update(zip(param_group_identifier_keys, (index + 1, 1, False, False), strict=True))
    optimizer = kind(groups, lr=0.001, foreach=False)
    wrapper = distrib_optimizer.DistributedOptimizer.__new__(distrib_optimizer.DistributedOptimizer)
    wrapper.optimizer = optimizer
    wrapper.config = OptimizerConfig()
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    wrapper.grad_scaler = None
    wrapper.model_param_group_index_map = {parameter: (0, i) for i, parameter in enumerate(parameters)}
    wrapper.gbuf_ranges = [
        {
            torch.float32: [
                {"param_map": {parameter: {"gbuf_world": range(parameter.numel())} for parameter in parameters}}
            ]
        }
    ]
    return wrapper, parameters


@pytest.mark.parametrize("kind", [torch.optim.Adam, torch.optim.AdamW])
def test_fresh_native_template_and_restore(kind):
    source, _ = make_optimizer(kind)
    common = source.state_dict()
    assert [group["step"] for group in common["optimizer"]["param_groups"]] == [0, 0]
    restored, parameters = make_optimizer(kind)
    restored.load_state_dict(copy.deepcopy(common))
    for parameter in parameters:
        state = restored.optimizer.state[parameter]
        assert state["step"].item() == 0
        assert state["exp_avg"].shape == parameter.shape
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
    assert all("step" not in group for group in restored.optimizer.param_groups)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    restored.optimizer.step()
    assert [restored.optimizer.state[p]["step"].item() for p in parameters] == [1, 1]


@pytest.mark.parametrize("kind", [torch.optim.Adam, torch.optim.AdamW])
def test_restored_steps_have_independent_storage_and_advance_once(kind):
    source, parameters = make_optimizer(kind)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    source.optimizer.step()
    common = source.state_dict()
    restored, restored_parameters = make_optimizer(kind)
    restored.load_state_dict(copy.deepcopy(common))
    steps = [restored.optimizer.state[p]["step"] for p in restored_parameters]
    assert steps[0].data_ptr() != steps[1].data_ptr()
    steps[0].add_(7)
    assert steps[1].item() == 1
    steps[0].sub_(7)
    for original, parameter in zip(parameters, restored_parameters, strict=True):
        state = restored.optimizer.state[parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            state[key].copy_(source.optimizer.state[original][key])
        parameter.grad = torch.ones_like(parameter)
    restored.optimizer.step()
    assert [step.item() for step in steps] == [2, 2]
    common = restored.state_dict()
    restored.load_state_dict(copy.deepcopy(common))
    assert [restored.optimizer.state[p]["step"].item() for p in restored_parameters] == [2, 2]
