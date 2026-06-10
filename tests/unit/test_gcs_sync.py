"""Unit tests for vime_plugins.checkpoint.gcs_sync (commands mocked; no real gsutil/gcloud)."""

from __future__ import annotations

import subprocess

import pytest

from vime_plugins.checkpoint import gcs_sync as mod


def test_sync_file_gs_uses_gcloud(monkeypatch):
    captured = {}
    monkeypatch.setattr(mod, "_run_cmd", lambda cmd, log_stdout: captured.setdefault("cmd", cmd))
    mod.sync_file_gs("/local/f", "gs://b/f")
    assert captured["cmd"] == ["gcloud", "storage", "cp", "/local/f", "gs://b/f"]


def test_sync_dir_gs_uses_rsync(monkeypatch):
    captured = {}
    monkeypatch.setattr(mod, "_run_cmd", lambda cmd, log_stdout: captured.setdefault("cmd", cmd))
    mod.sync_dir_gs("/local/d", "gs://b/d")
    assert captured["cmd"] == ["gsutil", "-m", "rsync", "-r", "/local/d", "gs://b/d"]


def test_sync_paths_gs_falls_back_to_slow(monkeypatch):
    calls = []

    def fake_run(cmd, log_stdout):
        calls.append(cmd)
        # the fast path is the first gsutil cp -r; make it fail once.
        if "cp" in cmd and "-r" in cmd and len(calls) == 1:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(mod, "_run_cmd", fake_run)
    mod.sync_paths_gs("/src", "gs://b/d", "*", fast_timeout=0)

    assert len(calls) == 2  # fast (failed) then slow (succeeded)
    assert calls[1][0] == "gsutil" and "parallel_thread_count" in " ".join(calls[1])


def test_check_file_exists_gs(monkeypatch):
    monkeypatch.setattr(mod, "_run_cmd", lambda cmd, log_stdout: None)
    assert mod.check_file_exists_gs("gs://b/f") is True

    def raise_err(cmd, log_stdout):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(mod, "_run_cmd", raise_err)
    assert mod.check_file_exists_gs("gs://b/missing") is False


def test_run_cmd_shell_quiet(monkeypatch):
    captured = {}

    def fake_check_call(cmd, stdout=None, shell=False, stderr=None):
        captured.update(cmd=cmd, shell=shell)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    mod._run_cmd(["gsutil", "ls", "gs://b"], log_stdout=False)
    assert captured["shell"] is True
    assert captured["cmd"].endswith("> /dev/null 2>&1")


def test_watch_and_sync_one_pass(monkeypatch, tmp_path):
    (tmp_path / "iter_0001").mkdir()
    synced = []
    monkeypatch.setattr(mod, "sync_dir_gs", lambda local, dest, log_stdout=False: synced.append((local, dest)))

    # break out of the infinite loop after the first sleep.
    def stop_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(mod.time, "sleep", stop_sleep)
    with pytest.raises(KeyboardInterrupt):
        mod.watch_and_sync(str(tmp_path), "gs://b/run", poll_interval=1)

    assert synced == [(str(tmp_path), "gs://b/run")]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
