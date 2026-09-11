"""CPU protocol checks; these do not replace distributed Ascend smoke runs."""

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parents[5]


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
    section = patch.split("+++ b/vllm/utils/local_checkpoint.py\n", 1)[1]
    source = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+")) + "\n"
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
