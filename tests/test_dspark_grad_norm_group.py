"""The DSpark draft's ``grad_norm_group`` tag must be accepted by Megatron's optimizer.

Megatron-core validates every ``grad_norm_group`` tag against
``megatron.core.optimizer.optimizer.SEPARATE_GRAD_NORM_GROUPS`` when it computes
the gradient norm, which only lists ``"mtp"``. An unregistered ``"dspark"`` tag
made the first optimizer step of online draft training raise
``ValueError: Unknown grad_norm_group 'dspark'``. The stub below copies that
contract: the validator reads the module-level tuple at call time.
"""

import sys
import types

import pytest

from vime.backends.megatron_utils.dspark.modeling import DSPARK_GRAD_NORM_GROUP, register_dspark_grad_norm_group

NUM_GPUS = 0

OPTIMIZER_MODULE = "megatron.core.optimizer.optimizer"


def _megatron_optimizer_stub(with_groups: bool) -> types.ModuleType:
    optimizer = types.ModuleType(OPTIMIZER_MODULE)
    if with_groups:
        optimizer.SEPARATE_GRAD_NORM_GROUPS = ("mtp",)

        def validate(grad_norm_group):
            if grad_norm_group not in optimizer.SEPARATE_GRAD_NORM_GROUPS:
                raise ValueError(f"Unknown grad_norm_group '{grad_norm_group}'.")

        optimizer._validate_grad_norm_group = validate
    return optimizer


@pytest.fixture
def megatron_optimizer(monkeypatch):
    def install(with_groups=True):
        optimizer = _megatron_optimizer_stub(with_groups)
        for name in ("megatron", "megatron.core", "megatron.core.optimizer"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, OPTIMIZER_MODULE, optimizer)
        return optimizer

    return install


def test_unregistered_dspark_tag_is_rejected(megatron_optimizer):
    optimizer = megatron_optimizer()
    with pytest.raises(ValueError, match="Unknown grad_norm_group 'dspark'"):
        optimizer._validate_grad_norm_group(DSPARK_GRAD_NORM_GROUP)


def test_register_adds_dspark_once_and_keeps_mtp(megatron_optimizer):
    optimizer = megatron_optimizer()
    register_dspark_grad_norm_group()
    register_dspark_grad_norm_group()
    assert optimizer.SEPARATE_GRAD_NORM_GROUPS == ("mtp", DSPARK_GRAD_NORM_GROUP)
    optimizer._validate_grad_norm_group(DSPARK_GRAD_NORM_GROUP)
    optimizer._validate_grad_norm_group("mtp")


def test_register_is_a_no_op_without_separate_groups(megatron_optimizer):
    optimizer = megatron_optimizer(with_groups=False)
    register_dspark_grad_norm_group()
    assert not hasattr(optimizer, "SEPARATE_GRAD_NORM_GROUPS")
