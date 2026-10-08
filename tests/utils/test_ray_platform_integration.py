from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _fake_platform(ray_ops, *, is_npu=False):
    return SimpleNamespace(ray=ray_ops, is_npu=is_npu)


@pytest.mark.parametrize(
    ("platform_name", "visible_env", "assigned_id", "expected_resource"),
    [
        ("cuda", "CUDA_VISIBLE_DEVICES", "7", "GPU"),
        ("npu", "ASCEND_RT_VISIBLE_DEVICES", "9", "NPU"),
    ],
)
def test_platform_ray_accelerator_ids_and_local_mapping(
    monkeypatch,
    platform_name,
    visible_env,
    assigned_id,
    expected_resource,
):
    import ray

    from vime.platforms import get_platform

    platform = get_platform(platform_name)
    monkeypatch.setenv(visible_env, f"unused,{assigned_id}")
    if platform_name == "cuda":
        monkeypatch.setattr(ray, "get_gpu_ids", lambda: [assigned_id])
    else:
        context = SimpleNamespace(get_accelerator_ids=lambda: {"NPU": [assigned_id]})
        monkeypatch.setattr(ray, "get_runtime_context", lambda: context)

    assert platform.ray.resource_name == expected_resource
    assert platform.ray.accelerator_ids() == [assigned_id]
    assert platform.ray.local_device_id() == 1


def test_placement_group_uses_platform_ray_resource_contract(monkeypatch):
    from vime.ray import placement_group as placement_group_module

    ray_ops = SimpleNamespace(
        resource_name="ACCEL",
        bundle_resources=Mock(side_effect=lambda: {"ACCEL": 1, "CPU": 1}),
        actor_options=Mock(side_effect=lambda fraction: {"resources": {"ACCEL": fraction}}),
    )
    monkeypatch.setattr(placement_group_module, "current_platform", lambda: _fake_platform(ray_ops, is_npu=True))

    created = {}

    class FakePlacementGroup:
        def ready(self):
            return "ready"

    def fake_placement_group(bundles, strategy):
        created["bundles"] = bundles
        created["strategy"] = strategy
        return FakePlacementGroup()

    actor_options = []

    class FakeInfoActor:
        def __init__(self, result):
            self.get_ip_and_gpu_id = SimpleNamespace(remote=lambda: result)

    class FakeInfoActorClass:
        results = iter([("10.0.0.1", "3"), ("10.0.0.1", "1")])

        @classmethod
        def options(cls, **options):
            actor_options.append(options)
            result = next(cls.results)
            return SimpleNamespace(remote=lambda: FakeInfoActor(result))

    monkeypatch.setattr(placement_group_module, "placement_group", fake_placement_group)
    monkeypatch.setattr(placement_group_module, "InfoActor", FakeInfoActorClass)
    wait_results = iter([([], ["ready"]), (["ready"], [])])
    monkeypatch.setattr(placement_group_module.ray, "wait", lambda *_args, **_kwargs: next(wait_results))
    monkeypatch.setattr(placement_group_module.ray, "cluster_resources", lambda: {"ACCEL": 2})
    monkeypatch.setattr(placement_group_module.ray, "available_resources", lambda: {"ACCEL": 2})
    monkeypatch.setattr(placement_group_module.ray, "get", lambda value: value)
    monkeypatch.setattr(placement_group_module.ray, "kill", lambda _actor: None)

    pg, reordered_indices, reordered_ids = placement_group_module._create_placement_group(2)

    assert isinstance(pg, FakePlacementGroup)
    assert created == {
        "bundles": [{"ACCEL": 1, "CPU": 1}, {"ACCEL": 1, "CPU": 1}],
        "strategy": "PACK",
    }
    assert reordered_indices == [1, 0]
    assert reordered_ids == ["1", "3"]
    assert [options["resources"] for options in actor_options] == [{"ACCEL": 1}, {"ACCEL": 1}]
    assert [options["num_gpus"] for options in actor_options] == [0, 0]
    assert ray_ops.bundle_resources.call_count == 2
    assert [entry.args for entry in ray_ops.actor_options.call_args_list] == [(1,), (1,)]


def test_ray_noset_visible_devices_keeps_ascend_entry():
    from vime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

    assert "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES" in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


def test_train_group_uses_npu_runtime_env_and_actor_resources(monkeypatch):
    from vime.ray import actor_group as actor_group_module

    runtime_env_inputs = []
    ray_ops = SimpleNamespace(
        train_runtime_env=lambda args, env: runtime_env_inputs.append((args, dict(env)))
        or {**env, "PLATFORM_ENV": "1"},
        actor_options=Mock(side_effect=lambda fraction: {"resources": {"ACCEL": fraction}}),
    )
    monkeypatch.setattr(actor_group_module, "current_platform", lambda: _fake_platform(ray_ops, is_npu=True))

    actor_module = types.ModuleType("vime.backends.megatron_utils.actor")
    actor_module.MegatronTrainRayActor = object
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.actor", actor_module)

    remote_declarations = []
    actor_allocations = []

    class FakeActorHandle:
        get_master_addr_and_port = SimpleNamespace(remote=lambda: ("127.0.0.1", 20000))
        init = SimpleNamespace(remote=lambda *_args, **_kwargs: 0)

    class FakeRemoteActor:
        def options(self, **options):
            actor_allocations.append(options)
            return self

        def remote(self, *_args):
            return FakeActorHandle()

    def fake_remote(**options):
        remote_declarations.append(options)
        return lambda _actor_impl: FakeRemoteActor()

    monkeypatch.setattr(actor_group_module.ray, "remote", fake_remote)
    monkeypatch.setattr(actor_group_module.ray, "get", lambda value: value)

    args = SimpleNamespace(
        train_env_vars={"USER_ENV": "yes"},
        offload_train=True,
        train_backend="megatron",
        colocate=False,
        use_routing_replay=False,
    )
    group = actor_group_module.RayTrainGroup(
        args=args,
        num_nodes=1,
        num_gpus_per_node=1,
        pg=(object(), [0], [7]),
        num_gpus_per_actor=0.4,
    )
    assert group.create() == [0]

    assert runtime_env_inputs[0][0] is args
    assert runtime_env_inputs[0][1]["USER_ENV"] == "yes"
    assert remote_declarations[0]["runtime_env"]["env_vars"]["PLATFORM_ENV"] == "1"
    assert remote_declarations[0]["num_gpus"] == 1
    assert actor_allocations[0]["resources"] == {"ACCEL": 0.4}
    assert actor_allocations[0]["num_gpus"] == 0
    ray_ops.actor_options.assert_called_once_with(0.4)


@pytest.mark.parametrize("is_npu", [False, True])
def test_rollout_engine_delegates_runtime_env_and_actor_options(monkeypatch, is_npu):
    from vime.backends.vllm_utils import engine_group as rollout_module

    runtime_env_inputs = []
    ray_ops = SimpleNamespace(
        rollout_runtime_env=lambda args, env: runtime_env_inputs.append((args, dict(env)))
        or {**env, "PLATFORM_ENV": "rollout"},
        actor_options=Mock(side_effect=lambda fraction: {"resources": {"ACCEL": fraction}}),
    )
    monkeypatch.setattr(rollout_module, "current_platform", lambda: _fake_platform(ray_ops, is_npu=is_npu))

    actor_options = []

    class FakeEngine:
        init = SimpleNamespace(remote=lambda **_kwargs: "init-ref")

    class FakeRemoteActor:
        def options(self, **options):
            actor_options.append(options)
            return self

        def remote(self, *_args, **_kwargs):
            return FakeEngine()

    monkeypatch.setattr(rollout_module.ray, "remote", lambda _actor_impl: FakeRemoteActor())
    monkeypatch.setattr(
        rollout_module,
        "_allocate_rollout_engine_addr_and_ports_normal",
        lambda **_kwargs: ({0: {}}, {0: 15001}),
    )

    args = SimpleNamespace(
        debug_train_only=False,
        num_gpus_per_node=8,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        rollout_external=False,
        colocate=True,
    )
    group = rollout_module.ServerGroup(
        args=args,
        pg=(object(), [0], [7]),
        all_engines=[None],
        num_gpus_per_engine=1,
        num_new_engines=1,
    )

    handles, cursors = group.start_engines()

    assert handles == ["init-ref"]
    assert cursors == {0: 15001}
    if is_npu:
        assert runtime_env_inputs[0][0] is args
        assert actor_options[0]["runtime_env"]["env_vars"]["PLATFORM_ENV"] == "rollout"
        assert actor_options[0]["resources"] == {"ACCEL": 0.2}
        assert actor_options[0]["num_gpus"] == 0
        ray_ops.actor_options.assert_called_once_with(0.2)
    else:
        assert runtime_env_inputs == []
        assert "resources" not in actor_options[0]
        assert "PLATFORM_ENV" not in actor_options[0]["runtime_env"]["env_vars"]
        assert actor_options[0]["num_gpus"] == 0.2
        ray_ops.actor_options.assert_not_called()

    # The main placement check must still run before an invalid slot is used.
    group.all_engines = [None]
    group.gpu_offset = 1
    with pytest.raises(ValueError, match="Invalid rollout server group GPU placement"):
        group.start_engines()


def test_train_actor_uses_npu_local_device_mapping(monkeypatch):
    from vime.ray import train_actor as train_actor_module

    local_device_id = Mock(return_value=5)
    monkeypatch.setattr(
        train_actor_module,
        "current_platform",
        lambda: _fake_platform(SimpleNamespace(local_device_id=local_device_id), is_npu=True),
    )

    assert train_actor_module.get_local_gpu_id() == 5
    local_device_id.assert_called_once_with()


def test_train_actor_keeps_main_cuda_local_device_mapping(monkeypatch):
    from vime.ray import train_actor as train_actor_module

    monkeypatch.setattr(
        train_actor_module,
        "current_platform",
        lambda: _fake_platform(SimpleNamespace(), is_npu=False),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")
    monkeypatch.setattr(
        train_actor_module.accelerator, "_ACCELERATOR", train_actor_module.accelerator.CUDAAccelerator()
    )
    monkeypatch.setattr(train_actor_module.ray, "get_gpu_ids", lambda: ["3"])

    assert train_actor_module.get_local_gpu_id() == 1
