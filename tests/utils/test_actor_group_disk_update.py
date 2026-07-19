"""CPU tests for full-disk version ownership in ``RayTrainGroup``."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def actor_group_module(monkeypatch):
    ray = types.ModuleType("ray")
    ray.get = lambda refs: refs
    ray.remote = lambda *args, **kwargs: lambda value: value
    ray.kill = lambda *args, **kwargs: None

    ray_util = types.ModuleType("ray.util")
    placement_group = types.ModuleType("ray.util.placement_group")
    placement_group.PlacementGroup = object
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.PlacementGroupSchedulingStrategy = object
    ray.util = ray_util

    ray_utils = types.ModuleType("vime.ray.utils")
    ray_utils.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = []
    ray_utils.add_default_ray_env_vars = lambda values=None: values or {}

    for name, module in {
        "ray": ray,
        "ray.util": ray_util,
        "ray.util.placement_group": placement_group,
        "ray.util.scheduling_strategies": scheduling,
        "vime.ray.utils": ray_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("vime.ray.actor_group", None)
    module = importlib.import_module("vime.ray.actor_group")
    yield module
    sys.modules.pop("vime.ray.actor_group", None)


class _RemoteMethod:
    def __init__(self, calls: list[dict[str, int]]) -> None:
        self.calls = calls

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class _Actor:
    def __init__(self, calls: list[dict[str, int]]) -> None:
        self.update_weights = _RemoteMethod(calls)


def _make_group(module, tmp_path: Path):
    group = module.RayTrainGroup.__new__(module.RayTrainGroup)
    group.args = SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport="disk",
        update_weight_disk_dir=str(tmp_path),
        release_train=False,
    )
    group.role = "actor"
    group._disk_weight_version = 0
    group._actor_handlers = []
    return group


def test_full_disk_group_commits_version_after_reload(actor_group_module, tmp_path: Path):
    group = _make_group(actor_group_module, tmp_path)
    actor_calls: list[dict[str, int]] = []
    reload_calls: list[tuple[Path, str]] = []
    group._actor_handlers = [_Actor(actor_calls)]
    group._reload_rollout_weights_from_disk = lambda path, version: reload_calls.append((path, version))

    group.update_weights()

    assert actor_calls == [{"weight_version": 1}]
    assert reload_calls == [(tmp_path / "weight_v000001", "1")]
    assert group._disk_weight_version == 1


def test_full_disk_group_retries_same_version_after_reload_failure(actor_group_module, tmp_path: Path):
    group = _make_group(actor_group_module, tmp_path)
    actor_calls: list[dict[str, int]] = []
    group._actor_handlers = [_Actor(actor_calls)]
    attempts = 0

    def reload_once_then_succeed(path, version):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("reload failed")

    group._reload_rollout_weights_from_disk = reload_once_then_succeed

    with pytest.raises(RuntimeError, match="reload failed"):
        group.update_weights()
    assert group._disk_weight_version == 0

    group.update_weights()

    assert actor_calls == [{"weight_version": 1}, {"weight_version": 1}]
    assert group._disk_weight_version == 1


def test_full_disk_group_retries_same_version_after_actor_failure(actor_group_module, tmp_path: Path):
    group = _make_group(actor_group_module, tmp_path)
    actor_calls: list[dict[str, int]] = []
    group._actor_handlers = [_Actor(actor_calls)]
    group._reload_rollout_weights_from_disk = lambda path, version: None
    original_get = actor_group_module.ray.get
    attempts = 0

    def fail_once(refs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("actor failed")
        return original_get(refs)

    actor_group_module.ray.get = fail_once

    with pytest.raises(RuntimeError, match="actor failed"):
        group.update_weights()
    assert group._disk_weight_version == 0

    group.update_weights()

    assert actor_calls == [{"weight_version": 1}, {"weight_version": 1}]
    assert group._disk_weight_version == 1
