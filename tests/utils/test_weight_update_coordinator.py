"""CPU unit tests for the shared weight update coordinator."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest

MODULE_PATH = "vime.backends.megatron_utils.update_weight.coordinator"
_STUBBED_MODULES = ("ray", "ray.actor", "vime.utils.distributed_utils", MODULE_PATH)


@pytest.fixture(scope="module")
def coordinator_module():
    saved = _unit_stubs.save_sys_modules(_STUBBED_MODULES)
    for name in _STUBBED_MODULES:
        sys.modules.pop(name, None)
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()
    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved)


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        self.events.append((self.name, args, kwargs))
        return self.name


@dataclass
class RecordingEngine:
    name: str
    events: list[tuple]
    pause_generation: RecordingRemoteMethod = field(init=False)
    flush_cache: RecordingRemoteMethod = field(init=False)
    post_process_weights: RecordingRemoteMethod = field(init=False)
    set_weight_version: RecordingRemoteMethod = field(init=False)
    continue_generation: RecordingRemoteMethod = field(init=False)

    def __post_init__(self) -> None:
        self.pause_generation = RecordingRemoteMethod(f"{self.name}.pause", self.events)
        self.flush_cache = RecordingRemoteMethod(f"{self.name}.flush", self.events)
        self.post_process_weights = RecordingRemoteMethod(f"{self.name}.post_process", self.events)
        self.set_weight_version = RecordingRemoteMethod(f"{self.name}.set_version", self.events)
        self.continue_generation = RecordingRemoteMethod(f"{self.name}.continue", self.events)


def _patch_runtime(
    monkeypatch,
    coordinator_module,
    *,
    rank: int,
    events: list[tuple],
    ray_get=None,
    broadcast=None,
):
    monkeypatch.setattr(coordinator_module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(
        coordinator_module.dist,
        "barrier",
        lambda *, group=None: events.append(("barrier", group)),
    )
    monkeypatch.setattr(
        coordinator_module.dist,
        "broadcast_object_list",
        broadcast
        or (
            lambda status, *, src, group: events.append(
                ("broadcast", src, group, status[0]),
            )
        ),
    )
    monkeypatch.setattr(coordinator_module, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(coordinator_module.ray, "get", ray_get or (lambda refs: refs))


@pytest.mark.unit
def test_success_orders_control_plane_and_returns_candidate(coordinator_module, monkeypatch):
    events: list[tuple] = []
    engines = [RecordingEngine("e0", events), RecordingEngine("e1", events)]
    _patch_runtime(monkeypatch, coordinator_module, rank=0, events=events)
    coordinator = coordinator_module.WeightUpdateCoordinator(
        engines,
        {"quant_method": "compressed-tensors"},
    )

    version = coordinator.run(
        current_version=7,
        transfer_target=lambda candidate: events.append(("target", candidate)),
        transfer_draft=lambda candidate: events.append(("draft", candidate)),
        commit=lambda: events.append(("commit",)),
    )

    assert version == 8
    names = [event[0] for event in events]
    assert names == [
        "e0.pause",
        "e1.pause",
        "e0.flush",
        "e1.flush",
        "e0.post_process",
        "e1.post_process",
        "broadcast",
        "target",
        "draft",
        "barrier",
        "e0.post_process",
        "e1.post_process",
        "commit",
        "e0.set_version",
        "e1.set_version",
        "e0.continue",
        "e1.continue",
        "broadcast",
    ]
    assert engines[0].post_process_weights.calls[0].kwargs == {
        "restore_weights_before_load": True,
        "post_process_quantization": False,
    }
    assert engines[0].post_process_weights.calls[1].kwargs == {
        "restore_weights_before_load": False,
        "post_process_quantization": True,
    }
    assert engines[0].set_weight_version.calls[0].args == ("8",)


@pytest.mark.unit
@pytest.mark.parametrize("failing_phase", ["target", "draft", "quant_post_process", "commit"])
def test_failure_before_publish_does_not_publish_or_resume(
    coordinator_module,
    monkeypatch,
    failing_phase,
):
    events: list[tuple] = []
    engines = [RecordingEngine("e0", events)]
    quant = {"quant_method": "compressed-tensors"} if failing_phase == "quant_post_process" else None

    def ray_get(refs):
        if (
            failing_phase == "quant_post_process"
            and refs == ["e0.post_process"]
            and len(engines[0].post_process_weights.calls) == 2
        ):
            raise RuntimeError("quant_post_process failed")
        return refs

    def phase(name):
        def run(*args):
            events.append((name, *args))
            if failing_phase == name:
                raise RuntimeError(f"{name} failed")

        return run

    _patch_runtime(monkeypatch, coordinator_module, rank=0, events=events, ray_get=ray_get)
    coordinator = coordinator_module.WeightUpdateCoordinator(engines, quant)

    with pytest.raises(RuntimeError, match=failing_phase):
        coordinator.run(
            current_version=3,
            transfer_target=phase("target"),
            transfer_draft=phase("draft"),
            commit=phase("commit"),
        )

    assert engines[0].set_weight_version.calls == []
    assert engines[0].continue_generation.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failing_call", "resumed"),
    [("e0.set_version", False), ("e0.continue", True)],
)
def test_failure_after_publish_restores_version_marker(
    coordinator_module,
    monkeypatch,
    failing_call,
    resumed,
):
    events: list[tuple] = []
    engines = [RecordingEngine("e0", events)]

    def ray_get(refs):
        if refs == [failing_call]:
            raise RuntimeError(f"{failing_call} failed")
        return refs

    _patch_runtime(monkeypatch, coordinator_module, rank=0, events=events, ray_get=ray_get)
    coordinator = coordinator_module.WeightUpdateCoordinator(engines, None)

    with pytest.raises(RuntimeError, match=failing_call):
        coordinator.run(current_version=4, transfer_target=lambda _candidate: None)

    # The candidate was published, then the committed marker was restored.
    assert [call.args for call in engines[0].set_weight_version.calls] == [
        ("5",),
        ("4",),
    ]
    if resumed:
        assert len(engines[0].pause_generation.calls) == 2
        assert len(engines[0].continue_generation.calls) == 1
    else:
        assert engines[0].continue_generation.calls == []


@pytest.mark.unit
def test_nonzero_rank_follows_rank_zero(coordinator_module, monkeypatch):
    # Success: nonzero ranks only run transfers and collectives, never engine RPCs.
    events: list[tuple] = []
    engines = [RecordingEngine("e0", events)]
    _patch_runtime(monkeypatch, coordinator_module, rank=1, events=events)
    coordinator = coordinator_module.WeightUpdateCoordinator(engines, None)

    version = coordinator.run(
        current_version=4,
        transfer_target=lambda candidate: events.append(("target", candidate)),
        transfer_draft=lambda candidate: events.append(("draft", candidate)),
        commit=lambda: events.append(("commit",)),
    )

    assert version == 5
    assert events == [
        ("broadcast", 0, "gloo", None),
        ("target", 5),
        ("draft", 5),
        ("barrier", "gloo"),
        ("broadcast", 0, "gloo", None),
    ]

    # Failure: a rank-0 control-plane error propagates through the status broadcast.
    fail_events: list[tuple] = []
    fail_engines = [RecordingEngine("e0", fail_events)]

    def broadcast(status, *, src, group):
        fail_events.append(("broadcast", src, group, status[0]))
        status[0] = "RuntimeError: pause failed"

    _patch_runtime(
        monkeypatch,
        coordinator_module,
        rank=1,
        events=fail_events,
        broadcast=broadcast,
    )
    failing = coordinator_module.WeightUpdateCoordinator(fail_engines, None)

    with pytest.raises(RuntimeError, match="quiesce.*pause failed"):
        failing.run(current_version=4, transfer_target=lambda _candidate: None)

    assert fail_engines[0].pause_generation.calls == []
