from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

from vime.platforms import current_platform, reset_platform_cache


@pytest.fixture(autouse=True)
def _reset_platform_selection():
    reset_platform_cache()
    yield
    reset_platform_cache()


def test_vime_platform_override_selects_cuda(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "cuda")

    platform = current_platform()

    assert platform.name == "cuda"
    assert platform.ray.resource_name == "GPU"
    assert platform.ray.visible_devices_env == "CUDA_VISIBLE_DEVICES"
    assert platform.checkpoint.default_megatron_to_hf_mode == "raw"


def test_vime_platform_override_selects_npu_without_vendor_import(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    before = {name for name in ("torch_npu", "vllm_ascend", "mindspeed") if name in sys.modules}

    platform = current_platform()

    after = {name for name in ("torch_npu", "vllm_ascend", "mindspeed") if name in sys.modules}
    assert platform.name == "npu"
    assert platform.ray.resource_name == "NPU"
    assert platform.checkpoint.default_megatron_to_hf_mode == "bridge"
    assert before == after


def test_unknown_explicit_platform_has_clear_error(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "not-registered")

    with pytest.raises(ValueError, match="Unknown Vime platform"):
        current_platform()


def test_npu_auto_detection_does_not_import_vendor_without_device_nodes(monkeypatch):
    from vime.platforms import npu

    monkeypatch.setattr(npu.os.path, "exists", lambda path: False)
    monkeypatch.setattr(npu, "glob", lambda pattern: [])

    def fail_import(name):
        raise AssertionError(f"unexpected import during negative NPU detection: {name}")

    monkeypatch.setattr(npu.importlib, "import_module", fail_import)

    assert npu.detect_npu() is False


def test_npu_safe_empty_cache_wraps_original_once(monkeypatch):
    from vime.platforms import npu

    calls = []

    def original_empty_cache():
        calls.append("original")
        raise RuntimeError("allocator is between offload states")

    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(empty_cache=original_empty_cache),
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(npu.importlib, "import_module", lambda name: fake_torch if name == "torch" else None)

    npu._install_safe_empty_cache()
    wrapped = fake_torch.npu.empty_cache
    wrapped()
    npu._install_safe_empty_cache()

    assert calls == ["original"]
    assert fake_torch.npu.empty_cache is wrapped
    assert fake_torch.cuda.empty_cache is wrapped


@pytest.mark.parametrize(
    ("name", "bundle", "actor_options"),
    [
        ("cuda", {"GPU": 2, "CPU": 3}, {"num_gpus": 0.4}),
        ("npu", {"NPU": 2, "CPU": 3}, {"resources": {"NPU": 0.4}}),
    ],
)
def test_ray_resource_contract(monkeypatch, name, bundle, actor_options):
    monkeypatch.setenv("VIME_PLATFORM", name)
    ray_spec = current_platform().ray

    assert ray_spec.bundle_resources(device_count=2, cpu_count=3) == bundle
    assert ray_spec.actor_options(0.4) == actor_options


def test_npu_runtime_env_is_scoped_to_npu_provider(monkeypatch, tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "python" / "site-packages" / "acl").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    args = Namespace(offload_train=True, train_backend="megatron", colocate=True)

    train_env = current_platform().ray.train_runtime_env(args, {"BASE": "1"})
    rollout_env = current_platform().ray.rollout_runtime_env(args, {"BASE": "1"})

    assert train_env["TMS_HOOK_MODE"] == "torch"
    assert train_env["TMS_REGION_TAG"] == "training"
    assert train_env["TMS_ENABLE_CPU_BACKUP"] == "1"
    assert train_env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert str(toolkit / "python" / "site-packages") in train_env["PYTHONPATH"]
    assert rollout_env["VLLM_USE_AOT_COMPILE"] == "0"
    assert rollout_env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"


def test_npu_vllm_env_replaces_cuda_and_rocm_visibility(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    platform = current_platform()

    env = platform.vllm.subprocess_env(
        {
            "KEEP": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "HIP_VISIBLE_DEVICES": "0,1",
        },
        visible_devices="4,5",
        colocate=True,
    )

    assert env["KEEP"] == "1"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "HIP_VISIBLE_DEVICES" not in env
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"


@pytest.mark.parametrize("name,backends", [("cuda", ("nccl", "ipc")), ("npu", ("hccl", "npu_ipc"))])
def test_weight_transfer_provider_selects_matching_backend_and_init_info(monkeypatch, name, backends):
    from vime.platforms import get_platform

    plugin_calls = []
    for colocate, backend in zip((False, True), backends, strict=True):
        if name == "cuda":
            module_name = f"vllm.distributed.weight_transfer.{backend}_engine"
            cls_name = "IPCTrainerInitInfo" if colocate else "NCCLTrainerInitInfo"
        else:
            module_name = f"vllm_ascend.distributed.weight_transfer.{backend}_engine"
            cls_name = "NPUIPCTrainerInitInfo" if colocate else "HCCLTrainerInitInfo"
        module = ModuleType(module_name)
        info_cls = type(cls_name, (SimpleNamespace,), {"backend": backend})
        setattr(module, cls_name, info_cls)
        monkeypatch.setitem(sys.modules, module_name, module)

        plugins = ModuleType("vllm.plugins")
        plugins.load_general_plugins = lambda: plugin_calls.append("load")
        monkeypatch.setitem(sys.modules, "vllm.plugins", plugins)
        monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

        ops = get_platform(name).weight_transfer
        info = ops.trainer_init_info(colocate=colocate, rank=3, packed=True)
        assert ops.backend("ipc" if colocate else "nccl") == info.backend == backend
        assert isinstance(info, info_cls)
        assert (info.rank, info.packed) == (3, True)

    assert plugin_calls == (["load", "load"] if name == "npu" else [])
    for backend in ("npu_ipc", "hccl", "custom"):
        assert ops.backend(backend) == backend


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("chained", [False, True])
def test_optimizer_state_initialization_reuses_megatron_callback(monkeypatch, platform, chained):
    monkeypatch.setenv("VIME_PLATFORM", platform)
    calls = []

    def init_state(optimizer, config):
        calls.append((optimizer, config))

    optimizers = [
        SimpleNamespace(optimizer=object(), config=object(), init_state_fn=init_state)
        for _ in range(2 if chained else 1)
    ]
    optimizer = SimpleNamespace(chained_optimizers=optimizers) if chained else optimizers[0]

    current_platform().megatron.initialize_optimizer_state(optimizer)

    expected = [(opt.optimizer, opt.config) for opt in optimizers] if platform == "npu" else []
    assert calls == expected


@pytest.mark.parametrize("empty_optimizer", [False, True])
def test_npu_optimizer_state_initialization_skips_missing_state_or_optimizer(monkeypatch, empty_optimizer):
    monkeypatch.setenv("VIME_PLATFORM", "npu")

    def unexpected_init(*args):
        raise AssertionError("an empty optimizer must not initialize state")

    optimizer = SimpleNamespace(
        optimizer=None if empty_optimizer else object(),
        config=object(),
        init_state_fn=unexpected_init if empty_optimizer else None,
    )

    current_platform().megatron.initialize_optimizer_state(optimizer)


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_eval_only_has_no_optimizer_state_to_initialize(monkeypatch, platform):
    monkeypatch.setenv("VIME_PLATFORM", platform)
    current_platform().megatron.initialize_optimizer_state(None)


def test_memory_utils_keep_main_accelerator_surface(monkeypatch):
    from vime.utils import memory_utils

    calls = []
    fake_accelerator = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
    )
    fake_torch = SimpleNamespace(_C=SimpleNamespace(_host_emptyCache=lambda: calls.append("empty_host_cache")))
    monkeypatch.setattr(memory_utils, "accelerator", fake_accelerator)
    monkeypatch.setattr(memory_utils, "torch", fake_torch)
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: calls.append("gc"))

    memory_utils.clear_memory(clear_host_memory=True)

    assert calls == ["synchronize", "gc", "empty_cache", "empty_host_cache"]
