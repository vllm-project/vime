"""Public model fallback, cache integrity, and unchanged GPU download policy."""

import json
import subprocess

import pytest
import torch
from safetensors.torch import save_file

from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils import hf_assets as mod
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0


def model(directory):
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text("{}")
    save_file({"a": torch.zeros(2)}, directory / "first.safetensors")
    save_file({"b": torch.ones(3)}, directory / "second.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}})
    )


def test_mirror_401_falls_back_in_fresh_process_without_changing_assets(monkeypatch, tmp_path):
    model(tmp_path)
    before = (tmp_path / "first.safetensors").read_bytes()
    calls = []
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")

    def run(command, *, env, check):
        calls.append((command, env.copy()))
        assert check
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    mod.download_public_model("Qwen/fixed-model", tmp_path, revision="fixed-revision")
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert calls[1][0][-2:] == ["--revision", "fixed-revision"]
    assert calls[0][1]["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert calls[1][1]["HF_ENDPOINT"] == "https://huggingface.co"
    assert calls[1][1]["HF_HUB_DISABLE_XET"] == "1"
    assert calls[1][1]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert (tmp_path / "first.safetensors").read_bytes() == before


def test_persistent_download_error_remains_a_failure_after_three_attempts(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: None)
    with pytest.raises(subprocess.CalledProcessError):
        mod.download_public_model("Qwen/model", tmp_path)
    assert len(calls) == 3


@pytest.mark.parametrize("damage", ["missing", "truncated", "invalid-header", "missing-config"])
def test_incomplete_cached_weights_cannot_be_accepted(tmp_path, damage):
    model(tmp_path)
    path = tmp_path / "second.safetensors"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == "invalid-header":
        path.write_bytes(b"\xff" * 16)
    else:
        (tmp_path / "config.json").unlink()
    with pytest.raises(ValueError):
        mod.validate_model_files(tmp_path)


def test_public_model_gpu_path_retains_original_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS["cuda"])
    commands = []
    monkeypatch.setattr(U, "exec_command", commands.append)
    U.hf_download_public_model("Qwen/model", dir_dst=tmp_path, revision="fixed")
    assert commands == [f"hf download Qwen/model --local-dir {tmp_path} --revision fixed"]
