import importlib.util
import sys
import types
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = True

    def __init__(self, *, named_tensors=None, flattened_tensor=None, metadata=None):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self._flattened_tensor = ("flattened", tuple(name for name, _ in named_tensors))
            self._metadata = tuple(name for name, _ in named_tensors)
            return

        self._flattened_tensor = flattened_tensor
        self._metadata = metadata

    def get_flattened_tensor(self):
        return self._flattened_tensor

    def get_metadata(self):
        return self._metadata


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(value, output_str):
        assert output_str is True
        return value


class _FakeRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return f"ref-{len(self.calls)}"


class _FakeEngine:
    def __init__(self):
        self.update_weights_from_tensor = _FakeRemoteMethod()


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
    torch_mod.uint8 = "uint8"
    torch_mod.distributed = dist_mod
    torch_mod.empty = lambda size, dtype, device: {"size": size, "dtype": dtype, "device": device}
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.cuda = types.SimpleNamespace(current_device=lambda: "cuda:0", ipc_collect=lambda: None)
    torch_mod.nn = types.SimpleNamespace(Module=object)

    ray_mod = types.ModuleType("ray")
    ray_mod.ObjectRef = object
    ray_actor_mod = types.ModuleType("ray.actor")
    ray_actor_mod.ActorHandle = object

    mpu_mod = types.ModuleType("megatron.core.mpu")
    megatron_mod = types.ModuleType("megatron")
    megatron_core_mod = types.ModuleType("megatron.core")
    megatron_core_mod.mpu = mpu_mod

    vllm_mod = types.ModuleType("vime.backends.megatron_utils.vllm")
    vllm_mod.FlattenedTensorBucket = _FakeFlattenedTensorBucket
    vllm_mod.MultiprocessingSerializer = _FakeMultiprocessingSerializer

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
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.vllm", vllm_mod)
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


def test_packed_colocated_bucket_rejects_mismatched_rank_metadata(monkeypatch):
    module, _ = _load_update_weight_module(monkeypatch)
    empty = {
        "names": [],
        "dtype_names": [],
        "shapes": [],
        "tensor_sizes": [],
        "ipc_handles": {"gpu-0": ("empty",)},
    }
    remote = {
        "names": ["expert.weight"],
        "dtype_names": ["bfloat16"],
        "shapes": [[4, 8]],
        "tensor_sizes": [64],
        "ipc_handles": {"gpu-1": ("remote",)},
    }

    with pytest.raises(ValueError, match="packed IPC metadata must match"):
        module._merge_ipc_update_infos([empty, remote])


def test_packed_colocated_bucket_merges_rank_handles(monkeypatch):
    module, _ = _load_update_weight_module(monkeypatch)
    first = {
        "names": ["shared.weight", "expert.weight"],
        "dtype_names": ["float16", "bfloat16"],
        "shapes": [[2, 2], [4, 8]],
        "tensor_sizes": [8, 64],
        "ipc_handles": {"gpu-0": ("first",)},
    }
    second = {
        **first,
        "ipc_handles": {"gpu-1": ("second",)},
    }

    assert module._merge_ipc_update_infos([first, second]) == {
        **first,
        "ipc_handles": {"gpu-0": ("first",), "gpu-1": ("second",)},
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
