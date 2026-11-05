"""Disk delta must update the same checkpoint shards the model loader reads."""

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import safetensors.numpy
import zstandard

from vime.utils.disk_delta import checksum, overwrite_encode

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
PATCHES = sorted((ROOT / "docker" / "patch").glob("*/vllm-pull_weights.patch")) + [
    ROOT / "docker/npu_patch/vllm.patch"
]


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "hf_checkpoint_reader", ROOT / "vime/backends/megatron_utils/hf_to_megatron/common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fail_training", [False, True])
def test_smoke_conversion_checkpoint_is_temporary_and_used_for_ref_load(monkeypatch, tmp_path, fail_training):
    monkeypatch.setitem(sys.modules, "vime.backends.megatron_utils.hf_to_megatron.common", _load_trainer())
    spec = importlib.util.spec_from_file_location("delta_disk_smoke", ROOT / "tests/test_delta_disk_weight_update.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setattr(smoke, "TEST_ROOT", str(tmp_path))
    monkeypatch.setattr(smoke.U, "exec_command", lambda command: None)
    monkeypatch.setattr(smoke.U, "get_default_wandb_args", lambda path: "")
    monkeypatch.setattr(smoke, "verify_checkpoints", lambda *args: None)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(key, raising=False)
    checkpoints = []
    rollouts = []

    def convert(**kwargs):
        root = Path(kwargs["dir_dst"])
        assert root.parent == tmp_path / "tmp"
        checkpoint = root / f"{kwargs['model_name']}_torch_dist"
        checkpoint.mkdir()
        checkpoints.append(checkpoint)

    def train(**kwargs):
        tokens = shlex.split(kwargs["train_args"])
        assert tokens[tokens.index("--ref-load") + 1] == str(checkpoints[0])
        assert checkpoints[0].is_dir()
        rollouts.append(tokens[tokens.index("--update-weight-delta-encoding") + 1])
        if fail_training:
            raise RuntimeError("training failed")

    monkeypatch.setattr(smoke.U, "convert_checkpoint", convert)
    monkeypatch.setattr(smoke.U, "execute_train", train)
    if fail_training:
        with pytest.raises(RuntimeError, match="training failed"):
            smoke.main()
    else:
        smoke.main()
    assert rollouts == (["xor"] if fail_training else ["xor", "overwrite"])
    assert len(checkpoints) == 1
    assert not checkpoints[0].parent.exists()


def _load_receiver(patch):
    # Test the exact standalone module shipped to vLLM in each patch bundle.
    text = patch.read_text()
    section = text.split("+++ b/vllm/utils/local_checkpoint.py\n", 1)[1]
    section = section.split("\ndiff --git ", 1)[0]
    source = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))
    module = ModuleType("local_checkpoint")
    exec(compile(source, str(patch), "exec"), module.__dict__)
    return module


def _write_checkpoint(path):
    path.mkdir()
    safetensors.numpy.save_file({"weight": np.ones((2, 2), dtype=np.float32)}, path / "model.safetensors")
    safetensors.numpy.save_file({"weight": np.zeros((2, 2), dtype=np.float32)}, path / "stale.safetensors")
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))


@pytest.fixture(params=[None, *PATCHES], ids=["trainer", *(p.parent.name for p in PATCHES)])
def reader(request, monkeypatch):
    if request.param is None:
        module = _load_trainer()
        return lambda path: module.SafetensorReader(path).get_tensor("weight").numpy()
    module = _load_receiver(request.param)
    # scandir/glob order is unspecified; force the stale file to be visited last.
    original_glob = module.glob.glob
    monkeypatch.setattr(module.glob, "glob", lambda pattern: sorted(original_glob(pattern)))

    def read_weight(path):
        filename, _, _ = module._tensor_locations(str(path))["weight"]
        return safetensors.numpy.load_file(filename)["weight"]

    return read_weight


def test_index_excludes_stale_shards(reader, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    np.testing.assert_array_equal(reader(checkpoint), np.ones((2, 2), dtype=np.float32))


def test_missing_indexed_shard_does_not_fall_back_to_stale_weights(reader, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        reader(checkpoint)


def test_unindexed_checkpoint_remains_supported(reader, tmp_path):
    safetensors.numpy.save_file({"weight": np.ones(4, dtype=np.float32)}, tmp_path / "model.safetensors")
    np.testing.assert_array_equal(reader(tmp_path), np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("patch", PATCHES, ids=[p.parent.name for p in PATCHES])
@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
@pytest.mark.parametrize("checksum_failure", [False, True])
def test_delta_updates_indexed_weights(patch, encoding, checksum_failure, tmp_path, monkeypatch):
    trainer, receiver = _load_trainer(), _load_receiver(patch)
    original_glob = receiver.glob.glob
    monkeypatch.setattr(receiver.glob, "glob", lambda pattern: sorted(original_glob(pattern)))
    base, local, stream = tmp_path / "base", tmp_path / "local", tmp_path / "stream"
    _write_checkpoint(base)
    stream.mkdir()
    receiver.pull_checkpoint(str(local), str(base), str(stream), 0)
    old = trainer.SafetensorReader(base).get_tensor("weight").numpy().view(np.uint8).reshape(-1)
    new = np.full((2, 2), 2, dtype=np.float32).view(np.uint8).reshape(-1)
    diff = new ^ old if encoding == "xor" else overwrite_encode(new, new != old)
    version = stream / "weight_v000001"
    version.mkdir()
    compressed = np.frombuffer(zstandard.ZstdCompressor().compress(diff), dtype=np.uint8)
    safetensors.numpy.save_file(
        {"weight": compressed}, version / "model.safetensors", metadata={"weight": checksum("adler32", new)}
    )
    (version / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": encoding,
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
                "weight_map": {"weight": "model.safetensors"},
            }
        )
    )
    if checksum_failure:
        # Corrupt a byte not changed by overwrite, so both encodings detect it.
        damaged = safetensors.numpy.load_file(local / "model.safetensors")["weight"]
        unchanged = np.flatnonzero(new == old)[0]
        damaged.view(np.uint8).reshape(-1)[unchanged] ^= 1
        safetensors.numpy.save_file({"weight": damaged}, local / "model.safetensors")
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            receiver.pull_checkpoint(str(local), str(base), str(stream), 1)
        assert receiver._read_applied_version(str(local)) == 0
    else:
        receiver.pull_checkpoint(str(local), str(base), str(stream), 1)
        receiver.pull_checkpoint(str(local), str(base), str(stream), 1)
        assert receiver._read_applied_version(str(local)) == 1
    # The delta's own checksum can pass while the indexed model stays stale.
    loaded = safetensors.numpy.load_file(local / "model.safetensors")["weight"]
    if not checksum_failure:
        np.testing.assert_array_equal(loaded.view(np.uint8).reshape(-1), new)
    np.testing.assert_array_equal(safetensors.numpy.load_file(local / "stale.safetensors")["weight"], 0)

    # A replacement trainer publishes a new full checkpoint; do not retry XOR
    # against files that were partially written before a checksum failure.
    full = stream / "weight_v000002"
    full.mkdir()
    reset = np.full((2, 2), 7, dtype=np.float32)
    safetensors.numpy.save_file({"weight": reset}, full / "model.safetensors")
    (full / "config.json").write_text("{}")
    receiver.pull_checkpoint(str(local), str(base), str(stream), 2)
    assert receiver._read_applied_version(str(local)) == 2
    np.testing.assert_array_equal(safetensors.numpy.load_file(local / "model.safetensors")["weight"], reset)
    assert not (local / "stale.safetensors").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
