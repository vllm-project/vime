import sys
from types import ModuleType, SimpleNamespace

import pytest

from vime.utils import accelerator

NUM_GPUS = 0


class FakeAccelerator(accelerator.Accelerator):
    name = "fake"
    device_type = "fake"
    communication_backend_name = "fake"

    def is_available(self):
        return True

    def device(self, index=None):
        return accelerator.torch.device("cpu")

    def device_name(self, index=None):
        return "cpu"

    def set_device(self, index):
        return None

    def current_device(self):
        return "cpu"

    def device_count(self):
        return 0

    def synchronize(self, device=None):
        return None

    def current_stream(self, device=None):
        return None

    def empty_cache(self):
        return None

    def mem_get_info(self, device=None):
        return 0, 0

    def memory_allocated(self, device=None):
        return 0

    def memory_reserved(self, device=None):
        return 0


@pytest.fixture(autouse=True)
def reset_accelerator_selection(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    patch_imported = accelerator._MUSA_TITAN_IMPORTED
    bootstrap_checked = accelerator._MUSA_BOOTSTRAP_CHECKED
    supa_imported = accelerator._SUPA_RUNTIME_IMPORTED
    supa_bootstrap_checked = accelerator._SUPA_BOOTSTRAP_CHECKED
    for name in (
        "VIME_ACCELERATOR",
        "MUSA_VISIBLE_DEVICES",
        "MUSA_TITAN_PATH",
        "CUDA_VISIBLE_DEVICES",
        "SUPA_VISIBLE_DEVICES",
        "BIREN_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    accelerator._MUSA_TITAN_IMPORTED = False
    accelerator._MUSA_BOOTSTRAP_CHECKED = False
    accelerator._SUPA_RUNTIME_IMPORTED = False
    accelerator._SUPA_BOOTSTRAP_CHECKED = False
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected
    accelerator._MUSA_TITAN_IMPORTED = patch_imported
    accelerator._MUSA_BOOTSTRAP_CHECKED = bootstrap_checked
    accelerator._SUPA_RUNTIME_IMPORTED = supa_imported
    accelerator._SUPA_BOOTSTRAP_CHECKED = supa_bootstrap_checked


@pytest.mark.unit
def test_cuda_selection_does_not_bootstrap_musa(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    monkeypatch.setattr(
        accelerator,
        "_import_musa_titan",
        lambda: pytest.fail("CUDA selection must not import musa_titan"),
    )

    assert accelerator.get_accelerator().name == "cuda"
    assert accelerator.process_group_backend() == "nccl"
    assert accelerator.visible_devices_env_key() == "CUDA_VISIBLE_DEVICES"


@pytest.mark.unit
def test_selected_musa_bootstraps_patch_once(monkeypatch):
    imports = []
    fake_musa = SimpleNamespace(is_available=lambda: True)

    def import_musa_titan():
        imports.append("musa_titan")
        monkeypatch.setattr(accelerator.torch, "musa", fake_musa, raising=False)
        return True

    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_import_musa_titan", import_musa_titan)

    assert imports == []
    assert accelerator.initialize_accelerator().name == "musa"
    assert accelerator.initialize_accelerator().name == "musa"
    assert imports == ["musa_titan"]


@pytest.mark.unit
def test_musa_titan_path_is_appended_once(monkeypatch, tmp_path):
    titan_path = str(tmp_path / "musa-titan")
    monkeypatch.setenv("MUSA_TITAN_PATH", titan_path)
    monkeypatch.setattr(accelerator.sys, "path", [])

    accelerator._append_musa_titan_path()
    accelerator._append_musa_titan_path()

    assert accelerator.sys.path == [titan_path]


@pytest.mark.unit
def test_import_musa_titan_prefers_apply_rl_patch(monkeypatch):
    imported = []
    callbacks = []
    module = SimpleNamespace(
        apply_rl_patch=lambda: callbacks.append("apply_rl_patch"),
        patch_after_import_torch=lambda: callbacks.append("patch_after_import_torch"),
    )

    def import_module(name):
        imported.append(name)
        return module

    monkeypatch.setattr(accelerator.importlib, "import_module", import_module)

    assert accelerator._import_musa_titan() is True
    assert imported == ["musa_titan"]
    assert callbacks == ["apply_rl_patch"]


@pytest.mark.unit
def test_import_musa_titan_falls_back_to_post_import_callback(monkeypatch):
    callbacks = []
    module = SimpleNamespace(patch_after_import_torch=lambda: callbacks.append("patch_after_import_torch"))
    monkeypatch.setattr(accelerator.importlib, "import_module", lambda name: module)

    assert accelerator._import_musa_titan() is True
    assert callbacks == ["patch_after_import_torch"]


@pytest.mark.unit
@pytest.mark.parametrize(("dependency", "expected"), [("musa_titan", False), ("torchada", True)])
def test_import_musa_titan_handles_missing_dependencies(monkeypatch, dependency, expected):
    def import_module(name):
        raise ModuleNotFoundError(f"No module named {dependency!r}", name=dependency)

    monkeypatch.setattr(accelerator.importlib, "import_module", import_module)

    if expected:
        with pytest.raises(RuntimeError, match="dependency 'torchada' is missing"):
            accelerator._import_musa_titan()
    else:
        assert accelerator._import_musa_titan() is False


@pytest.mark.unit
@pytest.mark.parametrize("has_apply_rl_patch", [True, False])
def test_musa_accelerator_post_import_torch_uses_compatible_callback(monkeypatch, has_apply_rl_patch):
    from vime.utils.accelerator import musa as musa_backend

    callbacks = []
    module = SimpleNamespace(
        apply_rl_patch=lambda: callbacks.append("apply_rl_patch"),
        patch_after_import_torch=lambda: callbacks.append("patch_after_import_torch"),
    )
    if not has_apply_rl_patch:
        del module.apply_rl_patch
    monkeypatch.setattr(musa_backend.importlib, "import_module", lambda name: module)

    accelerator.MUSAAccelerator().post_import_torch()

    assert callbacks == ["apply_rl_patch" if has_apply_rl_patch else "patch_after_import_torch"]


@pytest.mark.unit
def test_cpu_only_initialization_does_not_require_an_accelerator(monkeypatch):
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)

    assert accelerator.initialize_accelerator() is None


@pytest.mark.unit
def test_musa_backend_maps_devices_and_process_groups(monkeypatch):
    monkeypatch.setattr(accelerator.MUSAAccelerator, "is_available", lambda self: True)
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "2,5")
    accelerator.set_accelerator(accelerator.MUSAAccelerator())

    assert accelerator.visible_devices_env_key() == "MUSA_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1
    assert accelerator.process_group_backend() == "mccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,musa:mccl"
    assert accelerator.process_group_backend("gloo") == "gloo"


@pytest.mark.unit
def test_cuda_visible_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    accelerator.set_accelerator(FakeAccelerator())

    assert accelerator.resolve_visible_device_id(4) == 0
    assert accelerator.resolve_visible_device_id(1) == 1
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=4,6"):
        accelerator.resolve_visible_device_id(7)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")
    assert accelerator.resolve_visible_device_id("GPU-bbbb") == 1


@pytest.mark.unit
def test_registered_backend_can_be_selected(monkeypatch):
    class RegisteredAccelerator(FakeAccelerator):
        name = "registered"

    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    accelerator.register_accelerator("registered", RegisteredAccelerator, lambda: True, priority=300)

    assert accelerator.get_accelerator().name == "registered"


@pytest.mark.unit
def test_cuda_backend_uses_torch_cuda_namespace(monkeypatch):
    monkeypatch.setattr(accelerator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(accelerator.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(accelerator.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(accelerator.torch.cuda, "memory_allocated", lambda device=None: 123)
    backend = accelerator.CUDAAccelerator()

    assert backend.is_available()
    assert backend.device_name() == "cuda:1"
    assert backend.memory_allocated() == 123


@pytest.mark.unit
def test_routing_replay_uses_selected_backend_current_device(monkeypatch):
    from vime.utils import routing_replay

    transfers = []

    class FakeTopIndices:
        def is_pinned(self):
            return False

        def to(self, device, *, dtype, non_blocking):
            transfers.append((device, dtype, non_blocking))
            return self

    accelerator.set_accelerator(FakeAccelerator())
    monkeypatch.setattr(routing_replay.RoutingReplay, "all_routing_replays", [])
    replay = routing_replay.RoutingReplay()
    replay.top_indices_list.append(FakeTopIndices())

    replay.pop_forward()
    replay.pop_backward()
    assert transfers == [
        ("cpu", accelerator.torch.int32, False),
        ("cpu", accelerator.torch.int32, False),
    ]


@pytest.mark.unit
def test_musa_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(accelerator.torch, "musa", raising=False)

    assert accelerator.is_musa_available() is False


@pytest.mark.unit
def test_cuda_selection_does_not_bootstrap_supa(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    monkeypatch.setattr(
        accelerator,
        "_import_torch_supa",
        lambda: pytest.fail("CUDA selection must not import torch_supa"),
    )

    assert accelerator.get_accelerator().name == "cuda"
    assert accelerator.process_group_backend() == "nccl"
    assert accelerator.visible_devices_env_key() == "CUDA_VISIBLE_DEVICES"


@pytest.mark.unit
def test_selected_supa_bootstraps_runtime_once(monkeypatch):
    imports = []
    fake_supa = SimpleNamespace(is_available=lambda: True)

    def import_torch_supa():
        imports.append("torch_supa")
        monkeypatch.setattr(accelerator.torch, "supa", fake_supa, raising=False)
        return True

    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_import_torch_supa", import_torch_supa)

    assert imports == []
    assert accelerator.initialize_accelerator().name == "supa"
    assert accelerator.initialize_accelerator().name == "supa"
    assert imports == ["torch_supa"]


@pytest.mark.unit
def test_supa_backend_maps_devices_and_process_groups(monkeypatch):
    monkeypatch.setattr(accelerator.SUPAAccelerator, "is_available", lambda self: True)
    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "2,5")
    accelerator.set_accelerator(accelerator.SUPAAccelerator())

    assert accelerator.device_type() == "supa"
    assert accelerator.visible_devices_env_key() == "SUPA_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1
    assert accelerator.process_group_backend() == "bccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,supa:bccl"
    assert accelerator.process_group_backend("gloo") == "gloo"
    assert accelerator.distributed_device_id() is None


@pytest.mark.unit
def test_supa_backend_is_recognized_as_accelerator_backend():
    assert accelerator.is_accelerator_backend("bccl") is True
    assert accelerator.is_accelerator_backend("cpu:gloo,supa:bccl") is True
    assert accelerator.is_accelerator_backend("gloo") is False


@pytest.mark.unit
def test_supa_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(accelerator.torch, "supa", raising=False)

    assert accelerator.is_supa_available() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


@pytest.fixture
def no_local_backend(monkeypatch):
    monkeypatch.setattr(accelerator, "_register_builtin_backends", lambda: None)


class HookAccelerator(FakeAccelerator):
    ray_device_key = "CUSTOM"

    def __init__(self):
        raise AssertionError("Backend descriptions must not construct a device runtime")

    @classmethod
    def device_env_qualname(cls):
        return "test_accelerator_hooks"

    @classmethod
    def megatron_lifecycle_qualname(cls):
        return "test_accelerator_hooks"

    @classmethod
    def weight_transfer_backend(cls, transport):
        return f"custom_{transport}"

    @classmethod
    def weight_transfer_init_info_qualname(cls, transport):
        return "types.SimpleNamespace"


@pytest.mark.unit
@pytest.mark.parametrize("custom", [False, True])
def test_ray_resources_follow_backend_without_a_local_device(monkeypatch, no_local_backend, custom):
    import ray

    from vime.ray import resources

    if custom:
        accelerator.register_accelerator("custom", HookAccelerator, lambda: False)
        monkeypatch.setenv("VIME_ACCELERATOR", "custom")
    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [3], raising=False)
    context = SimpleNamespace(get_accelerator_ids=lambda: {"CUSTOM": ["5"]})
    monkeypatch.setattr(ray, "get_runtime_context", lambda: context, raising=False)

    assert resources.bundle(cpu_count=2) == {"CUSTOM" if custom else "GPU": 1, "CPU": 2}
    assert resources.actor_options(0.5) == (
        {"num_gpus": 0, "resources": {"CUSTOM": 0.5}} if custom else {"num_gpus": 0.5}
    )
    assert resources.accelerator_ids() == (["5"] if custom else ["3"])
    assert accelerator._ACCELERATOR is None


@pytest.mark.unit
def test_default_backend_hooks_preserve_the_callers_environment(no_local_backend):
    from vime.backends import megatron_lifecycle
    from vime.utils import device_env

    args = SimpleNamespace(offload_train=True)
    inherited = {"LD_PRELOAD": "caller.so", "TMS_INIT_ENABLE": "1", "CUDA_VISIBLE_DEVICES": "0,1"}
    expected = dict(inherited)
    for hook in (device_env.current().train_actor_env, device_env.current().engine_actor_env):
        assert hook(args, inherited) == expected
    assert device_env.current().serving_subprocess_env(inherited, visible_devices="2", colocate=True) == expected
    assert inherited == expected
    assert megatron_lifecycle.current() is megatron_lifecycle
    megatron_lifecycle.current().before_import()
    megatron_lifecycle.current().after_args(args)


@pytest.mark.unit
def test_backend_overrides_resolve_without_a_local_device(monkeypatch, no_local_backend):
    from vime.backends import megatron_lifecycle
    from vime.utils import device_env

    module = ModuleType("test_accelerator_hooks")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    accelerator.register_accelerator("custom", HookAccelerator, lambda: False)
    monkeypatch.setenv("VIME_ACCELERATOR", "custom")

    assert device_env.current() is module
    assert megatron_lifecycle.current() is module
    assert accelerator.weight_transfer_backend("distributed") == "custom_distributed"
    assert accelerator.weight_transfer_init_info("distributed", rank=3).rank == 3


@pytest.mark.unit
def test_function_factory_requires_runtime_selection_before_describing_backend(monkeypatch, no_local_backend):
    from vime.ray import resources

    calls = []

    def factory():
        calls.append("create")
        return FakeAccelerator()

    accelerator.register_accelerator("custom", factory, lambda: True)
    monkeypatch.setenv("VIME_ACCELERATOR", "custom")

    with pytest.raises(TypeError, match="custom.*requires class registration"):
        resources.actor_options(1)
    assert calls == []

    selected = accelerator.get_accelerator()
    assert accelerator.get_accelerator() is selected
    assert resources.actor_options(1) == {"num_gpus": 1}
    assert calls == ["create"]
