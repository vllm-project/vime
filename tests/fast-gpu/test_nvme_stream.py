"""NVMe numerical/checkpoint tests on the patched Megatron runtime.

Reuse the numerical test as a multi-GPU bucket smoke test (one reference per GPU):
  torchrun --standalone --nproc_per_node=8 -m pytest -q \
    tests/fast-gpu/test_nvme_stream.py -k bucket_fetch_step_matches
"""

import importlib.util
import json
import os
import sys
import types

import pytest
import torch

from vime_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Bucket, _Entry, _Stager


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    with torch.cuda.device(int(os.environ.get("LOCAL_RANK", "0"))):
        yield


@pytest.fixture
def bucket_factory(tmp_path, cuda_device):
    buckets = []

    def make(name, values, groups=(0,), optimizer_type=torch.optim.Adam, moment_dtype=torch.float32):
        params = [torch.nn.Parameter(values.clone().cuda()) for _ in groups]
        adam = optimizer_type([{"params": [p]} for p in params], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
        bucket = _Bucket(
            str(tmp_path / f"{name}.bin"),
            [_Entry(p, p, group) for p, group in zip(params, groups, strict=True)],
            adam,
            _Stager(1 << 20),
            {"main": torch.float32, "exp_avg": moment_dtype, "exp_avg_sq": moment_dtype},
        )
        buckets.append(bucket)
        return params, adam, bucket

    yield make
    for bucket in buckets:
        bucket.close()


def _store(buckets, resident=None):
    store = object.__new__(NVMeOptimizerStateStore)
    store._rank = store._instance = store.uid = 0
    store.buckets = buckets
    store.dtypes = buckets[0].dtypes if buckets else dict.fromkeys(("main", "exp_avg", "exp_avg_sq"), torch.float32)
    store._fp32_adam = resident
    store._allow_fresh_state = False
    return store


def _state_copy(optimizer, param):
    state = optimizer.state[param]
    return tuple(t.detach().clone() for t in (param, state["exp_avg"], state["exp_avg_sq"]))


def _assert_optimizer_equal(lhs, lhs_params, rhs, rhs_params):
    for index, (a, b) in enumerate(zip(lhs_params, rhs_params, strict=True)):
        for actual, expected in zip(_state_copy(lhs, a), _state_copy(rhs, b), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Fused Adam's group counter is authoritative, including after legacy loads.
        assert float(lhs.param_groups[index].get("step", lhs.state[a].get("step"))) == float(
            rhs.param_groups[index].get("step", rhs.state[b].get("step"))
        )


def test_bucket_fetch_step_matches_in_memory_adam(bucket_factory):
    initial = torch.linspace(-1.0, 1.0, 4096)
    params, streamed_opt, bucket = bucket_factory("stream", initial)
    streamed = params[0]
    streamed.grad = torch.linspace(0.1, 1.0, streamed.numel(), device=streamed.device)
    streamed_opt.step()
    reference = torch.nn.Parameter(initial.cuda())
    reference_opt = torch.optim.Adam([reference], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    reference.grad = streamed.grad.clone()
    for _ in range(2):
        reference_opt.step()
    bucket.flush()
    assert bucket.moments_ready
    bucket.fetch()
    streamed_opt.step()
    _assert_optimizer_equal(streamed_opt, params, reference_opt, [reference])


@pytest.mark.parametrize(
    "backend,groups,legacy",
    [
        pytest.param("torch", (0,), False, id="torch"),
        pytest.param("megatron", (0,), False, id="megatron"),
        pytest.param("megatron", (1,), True, id="legacy-group-1"),
        pytest.param("megatron", (1, 4), True, id="legacy-groups-1-4"),
    ],
)
def test_nvme_checkpoint_round_trip(bucket_factory, tmp_path, backend, groups, legacy):
    if backend == "megatron":
        from megatron.core.optimizer import Adam
    else:
        Adam = torch.optim.Adam
    initial = torch.linspace(-2.0, 2.0, 2048)
    params, adam, bucket = bucket_factory("live", initial, groups, Adam)
    restored_params, restored_adam, restored_bucket = bucket_factory("restored", initial, groups, Adam)
    for param in params:
        param.grad = torch.full_like(param, 0.25)
    adam.step()
    if legacy:
        # Distinct counters expose wrong aliases as well as out-of-range indexing.
        for index, group in enumerate(adam.param_groups):
            group["step"] = 2 + index * 3
    bucket.flush()
    store = _store([bucket])
    checkpoint = tmp_path / "checkpoint"
    store.save_to(str(checkpoint))
    if legacy:
        manifest_path = checkpoint / store.relative_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for meta in manifest["buckets"]:
            del meta["state_steps"]
        manifest_path.write_text(json.dumps(manifest))
    assert _store([restored_bucket]).load_from(str(checkpoint))
    bucket.fetch()
    restored_bucket.fetch()
    _assert_optimizer_equal(restored_adam, restored_params, adam, params)
    for original, loaded in zip(params, restored_params, strict=True):
        original.grad = torch.linspace(0.2, 0.9, original.numel(), device=original.device)
        loaded.grad = original.grad.clone()
    adam.step()
    restored_adam.step()
    _assert_optimizer_equal(restored_adam, restored_params, adam, params)


@pytest.mark.parametrize("missing_payload", [False, True])
def test_native_fp32_checkpoint_requires_resident_optimizer_state(tmp_path, cuda_device, monkeypatch, missing_payload):
    param = torch.nn.Parameter(torch.ones(32, device="cuda"))
    original = _store([], torch.optim.Adam([param], lr=1e-3))
    param.grad = torch.full_like(param, 0.25)
    original._fp32_adam.step()
    original.save_to(str(tmp_path))
    restored_param = torch.nn.Parameter(torch.ones_like(param))
    restored = _store([], torch.optim.Adam([restored_param], lr=1e-3))
    if missing_payload:
        (tmp_path / original.relative_dir / "fp32_resident_optimizer.pt").unlink()
        with pytest.raises(FileNotFoundError, match="fp32_resident_optimizer.pt"):
            restored.load_from(str(tmp_path))
    else:
        load_state_dict = restored._fp32_adam.load_state_dict

        def load_cpu_state(state_dict):
            for state in state_dict["state"].values():
                assert all(not torch.is_tensor(value) or value.device.type == "cpu" for value in state.values())
            return load_state_dict(state_dict)

        monkeypatch.setattr(restored._fp32_adam, "load_state_dict", load_cpu_state)
        assert restored.load_from(str(tmp_path))
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                restored._fp32_adam.state[restored_param][key],
                original._fp32_adam.state[param][key],
                atol=0,
                rtol=0,
            )


def test_bf16_moment_storage_round_trip(bucket_factory):
    params, adam, bucket = bucket_factory("bf16", torch.linspace(-1.0, 1.0, 4096), moment_dtype=torch.bfloat16)
    param = params[0]
    param.grad = torch.linspace(0.1, 1.0, param.numel(), device=param.device)
    adam.step()
    expected_main, expected_avg, expected_sq = _state_copy(adam, param)
    bucket.flush()
    bucket.fetch()
    torch.testing.assert_close(param, expected_main, atol=0, rtol=0)
    torch.testing.assert_close(adam.state[param]["exp_avg"], expected_avg, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(adam.state[param]["exp_avg_sq"], expected_sq, atol=2e-3, rtol=2e-3)


def _load_checkpoint_wrapper():
    megatron = types.ModuleType("megatron")
    training = types.ModuleType("megatron.training")
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    global_vars = types.ModuleType("megatron.training.global_vars")
    checkpointing.load_checkpoint = lambda *args, **kwargs: None
    checkpointing.save_checkpoint = lambda *args, **kwargs: None
    global_vars.get_args = lambda: None
    training.__path__ = []
    megatron.__path__ = []
    modules = {
        "megatron": megatron,
        "megatron.training": training,
        "megatron.training.checkpointing": checkpointing,
        "megatron.training.global_vars": global_vars,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    spec = importlib.util.spec_from_file_location(
        "vime_checkpoint_test_module",
        os.path.join(os.path.dirname(__file__), "../../vime/backends/megatron_utils/checkpoint.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for name, old in previous.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    return module


def test_checkpoint_wrapper_saves_streamed_state_before_tracker(tmp_path, monkeypatch):
    checkpointing = _load_checkpoint_wrapper()
    events = []

    class Store:
        def save_to(self, base):
            events.append(("nvme", base))

    args = types.SimpleNamespace(save=str(tmp_path), no_save_optim=False)
    optimizer = types.SimpleNamespace(_nvme_state_store=Store())
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpointing,
        "_save_checkpoint_megatron",
        lambda *a, **k: events.append(("megatron",)) or "saved",
    )

    result = checkpointing.save_checkpoint(3, None, optimizer, None)

    assert result == "saved"
    assert [event[0] for event in events] == ["nvme", "megatron"]
    assert events[0][1].endswith("iter_0000003")


@pytest.mark.parametrize(
    "mode,iteration,load_optimizer",
    [
        ("no-load-optim", 3, False),
        ("finetune", 0, False),
        ("release", 0, False),
        ("release-step-override", 0, False),
        ("resume-zero", 0, True),
        ("resume", 3, True),
    ],
)
def test_checkpoint_wrapper_load_modes(tmp_path, monkeypatch, mode, iteration, load_optimizer):
    checkpointing = _load_checkpoint_wrapper()
    tracker = "release" if mode.startswith("release") else str(3 if mode == "finetune" else iteration)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text(tracker)
    events = []

    class Store:
        def load_from(self, base):
            events.append(("load", base))
            return True

        def restore_main_to_model_params(self):
            events.append(("restore",))

    args = types.SimpleNamespace(
        load=str(tmp_path),
        no_load_optim=mode == "no-load-optim",
        finetune=mode == "finetune",
        ckpt_step=7 if mode == "release-step-override" else None,
    )
    optimizer = types.SimpleNamespace(_nvme_state_store=Store())
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)
    monkeypatch.setattr(
        checkpointing,
        "_load_checkpoint_megatron",
        lambda **kwargs: (iteration, 0),
    )

    result = checkpointing.load_checkpoint(None, optimizer, None, None)

    assert result == (iteration, 0)
    expected = [("load", str(tmp_path / f"iter_{iteration:07d}")), ("restore",)] if load_optimizer else []
    assert events == expected


@pytest.mark.parametrize("chained", [False, True])
def test_legacy_checkpoint_entrypoints_are_rejected(tmp_path, monkeypatch, chained):
    import megatron.core.optimizer.distrib_optimizer as distrib

    from vime_plugins.optimizers import nvme_stream as stream

    class DistributedOptimizer:
        is_stub_optimizer = False
        config = types.SimpleNamespace(defer_main_param_initialization=True)

    store = types.SimpleNamespace(dir=str(tmp_path), initialize_main_from_model_params=lambda: 0)
    monkeypatch.setattr(distrib, "DistributedOptimizer", DistributedOptimizer)
    monkeypatch.setattr(stream, "NVMeOptimizerStateStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(stream, "_purge_rank_dir", lambda root: None)
    dist_opt = DistributedOptimizer()
    optimizer = types.SimpleNamespace(chained_optimizers=[dist_opt]) if chained else dist_opt
    args = types.SimpleNamespace(
        offload_train_disk_dir=str(tmp_path),
        offload_train_disk_chunk_mb=1,
        stream_optimizer_state_moment_dtype="fp32",
    )
    stream.setup_optimizer_state_streaming(args, optimizer)
    for operation in ("save_parameter_state", "load_parameter_state"):
        with pytest.raises(RuntimeError, match="requires torch_dist"):
            getattr(optimizer, operation)(str(tmp_path / "legacy.pt"))


@pytest.mark.parametrize("flag", ["reset_optimizer_states", "load_main_params_from_ckpt"])
def test_streaming_rejects_incompatible_flags_before_purge(tmp_path, monkeypatch, flag):
    from vime_plugins.optimizers import nvme_stream as stream

    args = types.SimpleNamespace(_vime_nvme_role="critic", offload_train_disk_dir=str(tmp_path), **{flag: True})
    monkeypatch.setattr(stream, "_purge_rank_dir", lambda root: pytest.fail("must reject before clearing state"))
    with pytest.raises(AssertionError, match=flag.replace("_", "-")):
        stream.setup_optimizer_state_streaming(args, None)
