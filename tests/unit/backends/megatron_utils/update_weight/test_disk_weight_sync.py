"""CPU protocol checks; these do not replace distributed Ascend smoke runs."""

import importlib
import importlib.util
import json
import queue
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[5]


def _receiver_source(patch):
    section = patch.split("+++ b/vllm/utils/local_checkpoint.py\n", 1)[1]
    section = section.split("\ndiff --git", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+")) + "\n"


def test_receiver_source_ignores_following_file():
    patch = (ROOT / "docker/npu_patch/vllm.patch").read_text()
    appended = patch + "\ndiff --git a/other.py b/other.py\n+++ b/other.py\n@@ -0,0 +1 @@\n+invalid python!\n"
    assert _receiver_source(appended) == _receiver_source(patch)
    compile(_receiver_source(appended), "receiver.py", "exec")


@pytest.fixture
def modules(monkeypatch, tmp_path):
    # Load the updater package without importing the Megatron training runtime.
    package = types.ModuleType("_disk_sync_test")
    package.__path__ = [str(ROOT / "vime/backends/megatron_utils/update_weight")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded = []
    for name in ("update_weight_from_disk", "update_weight_from_disk_delta"):
        module = importlib.import_module(f"{package.__name__}.{name}")
        monkeypatch.setattr(module, "get_gloo_group", lambda: None)
        monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
        monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
        monkeypatch.setattr(module.dist, "all_gather_object", lambda out, value, **kwargs: out.__setitem__(0, value))
        monkeypatch.setattr(module.ray, "get", lambda values: values)
        loaded.append(module)
    monkeypatch.setattr(loaded[1], "NUM_WORKERS", 1)
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: empty(*args, **dict(kwargs, pin_memory=False)))
    iterator = SimpleNamespace(get_hf_weight_chunks=lambda weights, **kwargs: iter([list(weights.items())]))
    monkeypatch.setattr(loaded[0].HfWeightIteratorBase, "create", lambda **kwargs: iterator)

    # Exercise exactly the receiver shipped in the NPU patch.
    patch = (ROOT / "docker/npu_patch/vllm.patch").read_text()
    source = _receiver_source(patch)
    receiver_path = tmp_path / "receiver.py"
    receiver_path.write_text(source)
    spec = importlib.util.spec_from_file_location("disk_receiver", receiver_path)
    receiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(receiver)
    yield *loaded, receiver
    for name in list(sys.modules):
        if name.startswith("_disk_sync_test."):
            monkeypatch.delitem(sys.modules, name)


def make_args(tmp_path, encoding="xor"):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    return SimpleNamespace(
        hf_checkpoint=str(base),
        update_weight_disk_dir=str(tmp_path / "published"),
        update_weight_local_checkpoint_dir=str(tmp_path / "local"),
        update_weight_delta_encoding=encoding,
        update_weight_delta_checksum="adler32",
        custom_update_weight_post_write_path=None,
    )


def test_full_checkpoint_has_loadable_shards_and_index(modules, tmp_path):
    full, _, _ = modules
    args = make_args(tmp_path)
    weights = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    safetensors.torch.save_file(weights, Path(args.hf_checkpoint) / "model.safetensors")
    updater = full.UpdateWeightFromDisk(args, [], lambda: weights, model_name="qwen3", quantization_config=None)
    updater.update_weights()
    version = Path(args.update_weight_disk_dir) / "weight_v000001"
    index = json.loads((version / "model.safetensors.index.json").read_text())
    shard = safetensors.torch.load_file(version / index["weight_map"]["weight"])
    torch.testing.assert_close(shard["weight"], weights["weight"])
    assert index["metadata"]["total_size"] == 48
    assert (version / "config.json").exists()


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_delta_roundtrip_versions_and_repeated_pull(modules, tmp_path, encoding):
    _, delta, receiver = modules
    args = make_args(tmp_path, encoding)
    weights = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    safetensors.torch.save_file(weights, Path(args.hf_checkpoint) / "model.safetensors")
    updater = delta.UpdateWeightFromDiskDelta(args, [], lambda: weights, model_name="qwen3", quantization_config=None)
    updater.update_weights()  # Capture the same byte-exact HF base used by rollout.
    assert updater.weight_version == 0
    for version in (1, 2, 3):
        if version != 2:  # Include an unchanged-weight version.
            weights["weight"][0, 0] += version
        updater.weight_version = version
        updater._publish()
        for _ in range(2):  # Repeating XOR application must not corrupt bytes.
            receiver.pull_checkpoint(
                args.update_weight_local_checkpoint_dir, args.hf_checkpoint, args.update_weight_disk_dir, version
            )
        actual = safetensors.torch.load_file(Path(args.update_weight_local_checkpoint_dir) / "model.safetensors")
        np.testing.assert_array_equal(actual["weight"].numpy(), weights["weight"].numpy())


@pytest.mark.parametrize("failure", ["empty", "copyto", "submit", "device_copy"])
def test_pinned_buffer_returned_on_failure(modules, tmp_path, monkeypatch, failure):
    _, delta, _ = modules
    args = make_args(tmp_path)
    weights = {f"weight{i}": torch.arange(12, dtype=torch.float32) for i in range(3)}
    safetensors.torch.save_file(weights, Path(args.hf_checkpoint) / "model.safetensors")
    updater = delta.UpdateWeightFromDiskDelta(args, [], lambda: weights, model_name="qwen3", quantization_config=None)
    updater.update_weights()

    class BoundedWaitQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            # Turn a leaked-buffer deadlock into a bounded test failure.
            return super().get(block=block, timeout=2 if block and timeout is None else timeout)

    buffers = BoundedWaitQueue()
    buffer = torch.empty(48, dtype=torch.uint8)
    buffers.put(buffer)
    monkeypatch.setattr(delta, "_make_pinned_pool", lambda size: buffers)

    def fail(*args, **kwargs):
        raise MemoryError("injected buffer failure")

    if failure in ("empty", "copyto"):
        monkeypatch.setattr(delta.np, failure, fail)
    elif failure == "submit":
        monkeypatch.setattr(delta.ThreadPoolExecutor, "submit", fail)
    else:
        monkeypatch.setattr(torch.Tensor, "copy_", fail)
    with pytest.raises(MemoryError, match="injected buffer failure"):
        updater._encode_delta()
    assert buffers.qsize() == 1
    assert buffers.get_nowait() is buffer


@pytest.mark.parametrize("size,count", [(0, 0), (1 << 30, 8), (3 << 30, 2), (8 << 30, 1), ((8 << 30) + 1, 0)])
def test_pinned_pool_respects_budget(modules, monkeypatch, size, count):
    _, delta, _ = modules
    monkeypatch.setattr(delta, "NUM_WORKERS", 16)
    allocations = []

    def allocate(nbytes, **kwargs):
        allocations.append(nbytes)
        return object()

    monkeypatch.setattr(delta.torch, "empty", allocate)
    buffers = delta._make_pinned_pool(size)
    assert buffers.qsize() == count
    assert allocations == [size] * count
    assert sum(allocations) <= 8 << 30


@pytest.mark.parametrize("error", [RuntimeError, MemoryError])
def test_pinned_pool_releases_partial_allocations(modules, monkeypatch, error):
    _, delta, _ = modules
    references = []

    class Buffer:
        pass

    def allocate(*args, **kwargs):
        if references:
            raise error("allocation failed")
        buffer = Buffer()
        references.append(weakref.ref(buffer))
        return buffer

    monkeypatch.setattr(delta.torch, "empty", allocate)
    assert delta._make_pinned_pool(1024).empty()
    assert references[0]() is None
