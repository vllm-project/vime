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


def test_http_retries_keep_reachable_mirror_and_original_assets(monkeypatch, tmp_path):
    model(tmp_path)
    before = (tmp_path / "first.safetensors").read_bytes()
    calls = []
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setenv("HTTPS_PROXY", "http://ci-proxy:3128")
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

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
    for _, env in calls:
        assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert env["HF_HUB_DISABLE_XET"] == "1"
        assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
        assert env["HTTPS_PROXY"] == "http://ci-proxy:3128"
    assert "HF_HUB_DISABLE_XET" not in mod.os.environ
    assert (tmp_path / "first.safetensors").read_bytes() == before


@pytest.mark.parametrize("endpoint", [None, "https://models.internal.example"])
def test_http_download_preserves_explicit_endpoint_and_timeout_policy(monkeypatch, tmp_path, endpoint):
    model(tmp_path)
    if endpoint is None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "7")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda command, **kwargs: calls.append(kwargs["env"]))
    mod.download_public_model("Qwen/fixed-model", tmp_path)
    assert len(calls) == 1
    assert calls[0].get("HF_ENDPOINT") == endpoint
    assert calls[0]["HF_HUB_DISABLE_XET"] == "1"
    assert calls[0]["HF_HUB_ETAG_TIMEOUT"] == "7"
    assert calls[0]["HF_HUB_DOWNLOAD_TIMEOUT"] == "90"


def test_http_timeout_defaults_are_finite(monkeypatch, tmp_path):
    model(tmp_path)
    for key in ("HF_HUB_ETAG_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(mod.subprocess, "run", lambda command, **kwargs: calls.append(kwargs["env"]))
    mod.download_public_model("Qwen/fixed-model", tmp_path)
    assert calls[0]["HF_HUB_ETAG_TIMEOUT"] == "15"
    assert calls[0]["HF_HUB_DOWNLOAD_TIMEOUT"] == "60"


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
