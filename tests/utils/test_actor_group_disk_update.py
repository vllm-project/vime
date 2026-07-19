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


class _NamedRemoteMethod:
    def __init__(self, name: str, calls: list[str], result=None) -> None:
        self.name = name
        self.calls = calls
        self.result = result if result is not None else name
        self.kwargs: list[dict] = []

    def remote(self, *args, **kwargs):
        self.calls.append(self.name)
        self.kwargs.append(kwargs)
        return self.result


class _RolloutEngine:
    def __init__(self, calls: list[str], pull_result=None) -> None:
        self.pull_weights = _NamedRemoteMethod(
            "pull",
            calls,
            pull_result or {"success": True, "local_checkpoint_dir": "/remote/local"},
        )
        self.pause_generation = _NamedRemoteMethod("pause", calls)
        self.flush_cache = _NamedRemoteMethod("flush", calls)
        self.update_weights_from_disk = _NamedRemoteMethod("reload", calls)
        self.continue_generation = _NamedRemoteMethod("continue", calls)


class _RolloutManager:
    def __init__(self, engine) -> None:
        engines = engine if isinstance(engine, list) else [engine]
        self.get_updatable_engines_and_lock = _NamedRemoteMethod(
            "get_engines",
            [],
            (engines, None, 0, [], []),
        )


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


@pytest.mark.parametrize(
    ("failed_call", "expected_calls"),
    [
        ("pause", ["pause", "continue"]),
        ("flush", ["pause", "flush", "continue"]),
        ("reload", ["pause", "flush", "reload", "continue"]),
    ],
)
def test_disk_reload_resumes_engines_after_failure(
    actor_group_module,
    tmp_path: Path,
    failed_call: str,
    expected_calls: list[str],
):
    group = _make_group(actor_group_module, tmp_path)
    group.args.offload_rollout = False
    group.args.update_weight_local_checkpoint_dir = None
    group.args.update_weight_disk_keep_files = True
    group.args.ci_test = False
    calls: list[str] = []
    engine = _RolloutEngine(calls)
    group._rollout_manager = _RolloutManager(engine)

    def ray_get(refs):
        if refs == [failed_call]:
            raise RuntimeError(f"{failed_call} failed")
        return refs

    actor_group_module.ray.get = ray_get

    with pytest.raises(RuntimeError, match=rf"{failed_call} failed"):
        group._reload_rollout_weights_from_disk(tmp_path / "weight_v000001", "1")

    assert calls == expected_calls


def test_disk_reload_uses_server_returned_local_path(actor_group_module, tmp_path: Path):
    group = _make_group(actor_group_module, tmp_path)
    group.args.offload_rollout = False
    group.args.update_weight_local_checkpoint_dir = "/trainer/local"
    group.args.update_weight_disk_keep_files = True
    group.args.ci_test = False
    calls: list[str] = []
    engine = _RolloutEngine(calls, {"success": True, "local_checkpoint_dir": "/remote/local"})
    group._rollout_manager = _RolloutManager(engine)

    group._reload_rollout_weights_from_disk(tmp_path / "weight_v000001", "1")

    assert calls == ["pull", "pause", "flush", "reload", "continue"]
    assert engine.pull_weights.kwargs == [{"target_version": 1}]
    assert engine.update_weights_from_disk.kwargs[0]["model_path"] == "/remote/local"


def test_disk_reload_fans_out_server_returned_local_paths(actor_group_module, tmp_path: Path):
    group = _make_group(actor_group_module, tmp_path)
    group.args.offload_rollout = False
    group.args.update_weight_local_checkpoint_dir = "/trainer/local"
    group.args.update_weight_disk_keep_files = True
    group.args.ci_test = False
    calls: list[str] = []
    engines = [
        _RolloutEngine(calls, {"success": True, "local_checkpoint_dir": "/remote/engine-0"}),
        _RolloutEngine(calls, {"success": True, "local_checkpoint_dir": "/remote/engine-1"}),
    ]
    group._rollout_manager = _RolloutManager(engines)

    group._reload_rollout_weights_from_disk(tmp_path / "weight_v000001", "1")

    assert [engine.pull_weights.kwargs for engine in engines] == [
        [{"target_version": 1}],
        [{"target_version": 1}],
    ]
    assert [engine.update_weights_from_disk.kwargs[0]["model_path"] for engine in engines] == [
        "/remote/engine-0",
        "/remote/engine-1",
    ]
