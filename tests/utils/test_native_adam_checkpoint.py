"""Exercise MCore's native Torch Adam checkpoint path with real CPU AdamW state."""

import copy
import types
from types import SimpleNamespace

import pytest
import torch

distributed_optimizer = pytest.importorskip("megatron.core.optimizer.distrib_optimizer")


def make_optimizer():
    parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0])), torch.nn.Parameter(torch.tensor([3.0, 4.0]))]
    group = {key: False if key.startswith("is_") else 1.0 for key in distributed_optimizer.param_group_identifier_keys}
    optimizer = torch.optim.AdamW([{"params": parameters, **group}], lr=0.01)
    wrapper = SimpleNamespace(
        optimizer=optimizer,
        grad_scaler=None,
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        config=SimpleNamespace(fp16=False, use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        model_param_group_index_map={parameter: (0, index) for index, parameter in enumerate(parameters)},
    )
    wrapper.state_dict = types.MethodType(distributed_optimizer.DistributedOptimizer.state_dict, wrapper)
    wrapper.load_state_dict = types.MethodType(distributed_optimizer.DistributedOptimizer.load_state_dict, wrapper)
    wrapper._set_main_param_and_optimizer_states = types.MethodType(
        distributed_optimizer.DistributedOptimizer._set_main_param_and_optimizer_states, wrapper
    )
    return parameters, optimizer, wrapper


def update(parameters, optimizer):
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, index + 0.5)
    optimizer.step()


def test_fresh_native_optimizer_has_valid_checkpoint_template(monkeypatch):
    monkeypatch.setattr(distributed_optimizer, "HAVE_APEX_OR_TE", False)
    _, optimizer, wrapper = make_optimizer()
    state = wrapper.state_dict()
    assert not optimizer.state  # Template inspection must not run an optimizer update.
    assert all(group["step"] == 0 for group in state["optimizer"]["param_groups"])


def test_restored_native_adam_steps_are_independent_and_update_once(monkeypatch):
    monkeypatch.setattr(distributed_optimizer, "HAVE_APEX_OR_TE", False)
    parameters, optimizer, wrapper = make_optimizer()
    for _ in range(4):
        update(parameters, optimizer)
    saved_common = copy.deepcopy(wrapper.state_dict())
    saved_tensors = copy.deepcopy(optimizer.state_dict())
    restored_parameters, restored, restored_wrapper = make_optimizer()
    # Preallocate CPU state. Distributed checkpoint loading separately allocates
    # GPU shards; this unit test targets the scalar/non-parameter load method.
    update(restored_parameters, restored)
    restored_wrapper.load_state_dict(saved_common)
    with torch.no_grad():
        for index, (old, new) in enumerate(zip(parameters, restored_parameters, strict=True)):
            # DP-reshardable parameter shards exclude step: common state above
            # already restored it. Exercise the real shard setter as well.
            restored_wrapper._set_main_param_and_optimizer_states(
                new, {"param": old, **{k: saved_tensors["state"][index][k] for k in ["exp_avg", "exp_avg_sq"]}}
            )
    steps = [restored.state[p]["step"] for p in restored_parameters]
    assert len({step.data_ptr() for step in steps}) == len(steps)
    assert [step.item() for step in steps] == [4.0, 4.0]
    assert all("step" not in group for group in restored.param_groups)
    update(parameters, optimizer)
    update(restored_parameters, restored)
    for old, new in zip(parameters, restored_parameters, strict=True):
        assert torch.equal(old, new)
        for field in ["step", "exp_avg", "exp_avg_sq"]:
            assert torch.equal(optimizer.state[old][field], restored.state[new][field])
    assert [step.item() for step in steps] == [5.0, 5.0]


def test_inconsistent_native_steps_still_fail(monkeypatch):
    monkeypatch.setattr(distributed_optimizer, "HAVE_APEX_OR_TE", False)
    parameters, optimizer, wrapper = make_optimizer()
    update(parameters, optimizer)
    optimizer.state[parameters[0]]["step"].fill_(2)
    with pytest.raises(AssertionError):
        wrapper.state_dict()


def test_missing_parameter_moment_still_fails(monkeypatch):
    monkeypatch.setattr(distributed_optimizer, "HAVE_APEX_OR_TE", False)
    parameters, optimizer, wrapper = make_optimizer()
    update(parameters, optimizer)
    with torch.no_grad(), pytest.raises(KeyError, match="exp_avg"):
        wrapper._set_main_param_and_optimizer_states(parameters[0], {"param": parameters[0]})
