import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return f"ref-{len(self.calls)}"


class _FakeEngine:
    def __init__(self):
        self.update_weights = _FakeRemoteMethod()


def _install_fake_deps(monkeypatch):
    dist_state = types.SimpleNamespace(rank=0, world_size=2, gathered=None, local_object=None)

    vime_pkg = types.ModuleType("vime")
    vime_pkg.__path__ = [str(REPO_ROOT / "vime")]
    vime_backends_pkg = types.ModuleType("vime.backends")
    vime_backends_pkg.__path__ = [str(REPO_ROOT / "vime" / "backends")]
    megatron_utils_pkg = types.ModuleType("vime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(REPO_ROOT / "vime" / "backends" / "megatron_utils")]
    update_weight_pkg = types.ModuleType("vime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(REPO_ROOT / "vime" / "backends" / "megatron_utils" / "update_weight")]
    vime_utils_pkg = types.ModuleType("vime.utils")
    vime_utils_pkg.__path__ = [str(REPO_ROOT / "vime" / "utils")]

    dist_mod = types.ModuleType("torch.distributed")

    def gather_object(obj, object_gather_list, dst, group):
        dist_state.local_object = obj
        if object_gather_list is not None:
            object_gather_list[:] = dist_state.gathered(obj)

    dist_mod.get_rank = lambda: dist_state.rank
    dist_mod.get_world_size = lambda group=None: dist_state.world_size
    dist_mod.gather_object = gather_object

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.uint8 = "uint8"
    torch_mod.distributed = dist_mod
    torch_mod.empty = lambda size, dtype, device: {"size": size, "dtype": dtype, "device": device}
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.cuda = types.SimpleNamespace(
        current_device=lambda: 0,
        get_device_properties=lambda _device: types.SimpleNamespace(uuid="gpu-0"),
        ipc_collect=lambda: None,
    )
    torch_mod.nn = types.SimpleNamespace(Module=object)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod

    update_weight_common_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.common")
    update_weight_common_mod.HfWeightSource = object
    update_weight_common_mod.VimeRayWeightSyncClient = object
    update_weight_common_mod.create_nccl_trainer = lambda *args, **kwargs: None

    megatron_to_hf_mod = types.ModuleType("vime.backends.megatron_utils.megatron_to_hf")
    megatron_to_hf_mod.convert_to_hf = lambda *args, **kwargs: []

    expert_routing_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.expert_routing")
    expert_routing_mod.configure_expert_routing = lambda *args, **kwargs: (None, [])

    hf_weight_iterator_base_mod = types.ModuleType(
        "vime.backends.megatron_utils.update_weight.hf_weight_iterator_base"
    )
    hf_weight_iterator_base_mod.HfWeightIteratorBase = types.SimpleNamespace(create=lambda *args, **kwargs: None)

    vime_utils_types_mod = types.ModuleType("vime.utils.types")
    vime_utils_types_mod.ParamInfo = type("ParamInfo", (), {})

    distributed_utils_mod = types.ModuleType("vime.utils.distributed_utils")
    distributed_utils_mod.get_gloo_group = lambda: object()

    update_from_distributed_mod = types.ModuleType(
        "vime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )
    update_from_distributed_mod.connect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.disconnect_rollout_engines_from_distributed = lambda *args, **kwargs: None
    update_from_distributed_mod.post_process_weights = lambda *args, **kwargs: None
    update_from_distributed_mod.update_weights_from_distributed = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "vime", vime_pkg)
    monkeypatch.setitem(sys.modules, "vime.backends", vime_backends_pkg)
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils", megatron_utils_pkg)
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.update_weight", update_weight_pkg)
    monkeypatch.setitem(sys.modules, "vime.utils", vime_utils_pkg)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_mod)
    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor_mod)
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu_mod)
    monkeypatch.setitem(
        sys.modules,
        "vime.backends.megatron_utils.update_weight.common",
        update_weight_common_mod,
    )
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.megatron_to_hf", megatron_to_hf_mod)
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.update_weight.expert_routing", expert_routing_mod)
    monkeypatch.setitem(
        sys.modules,
        "vime.backends.megatron_utils.update_weight.hf_weight_iterator_base",
        hf_weight_iterator_base_mod,
    )
    monkeypatch.setitem(sys.modules, "vime.utils.types", vime_utils_types_mod)
    monkeypatch.setitem(sys.modules, "vime.utils.distributed_utils", distributed_utils_mod)
    monkeypatch.setitem(
        sys.modules,
        "vime.backends.megatron_utils.update_weight.update_weight_from_distributed",
        update_from_distributed_mod,
    )

    return dist_state


def _load_update_weight_module(monkeypatch):
    dist_state = _install_fake_deps(monkeypatch)

    module_name = "vime.backends.megatron_utils.update_weight.update_weight_from_tensor"
    sys.modules.pop(module_name, None)
    module_path = REPO_ROOT / "vime" / "backends" / "megatron_utils" / "update_weight" / "update_weight_from_tensor.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, dist_state


def test_empty_colocated_bucket_still_participates_in_gather(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    dist_state.gathered = lambda local: [local, local]
    engine = _FakeEngine()

    refs, long_lived_tensor = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
    )

    assert dist_state.local_object["names"] == []
    assert refs == []
    assert long_lived_tensor is None
    assert engine.update_weights.calls == []


def test_source_rank_marks_empty_colocated_bucket_gpu(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    remote_info = {
        "names": ["expert.weight"],
        "dtype_names": ["bfloat16"],
        "shapes": [[4, 8]],
        "tensor_sizes": [64],
        "ipc_handles": {"gpu-1": ("remote",)},
    }
    dist_state.gathered = lambda local: [local, remote_info]
    engine = _FakeEngine()

    refs, long_lived_tensor = module._send_to_colocated_engine(
        [],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
    )

    assert refs == ["ref-1"]
    assert long_lived_tensor is None
    assert engine.update_weights.calls == [(([None, remote_info],), {})]


def test_source_rank_sends_different_expert_metadata_as_separate_updates(monkeypatch):
    module, dist_state = _load_update_weight_module(monkeypatch)
    local_info = {
        "names": ["experts.0.weight"],
        "dtype_names": ["bfloat16"],
        "shapes": [[4, 8]],
        "tensor_sizes": [64],
        "ipc_handles": {"gpu-0": ("local",)},
    }
    remote_info = {
        **local_info,
        "names": ["experts.1.weight"],
        "ipc_handles": {"gpu-1": ("remote",)},
    }
    dist_state.gathered = lambda local: [local, remote_info]
    engine = _FakeEngine()

    monkeypatch.setattr(module, "_build_packed_ipc_update_info", lambda _tensors: (local_info, "packed"))
    refs, long_lived_tensor = module._send_to_colocated_engine(
        [("experts.0.weight", object())],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group=object(),
    )

    assert refs == ["ref-1"]
    assert long_lived_tensor == "packed"
    assert engine.update_weights.calls == [(([local_info, remote_info],), {})]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
