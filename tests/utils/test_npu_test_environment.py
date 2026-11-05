"""Resource logging must not dump credentials or mistake resource labels for devices."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

NUM_GPUS = 0


def test_report_records_observations_without_credentials(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "ci/npu_test_environment.py"
    spec = importlib.util.spec_from_file_location("npu_environment_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("HF_TOKEN", "secret-not-for-logs")
    monkeypatch.setenv("BUILDKITE_AGENT_ACCESS_TOKEN", "another-secret-not-for-logs")
    monkeypatch.setenv("VIME_TEST_RESOURCE_CLASS", "npu-4")
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "12,13,14,15")
    fake_torch = ModuleType("torch")
    fake_torch.npu = SimpleNamespace(device_count=lambda: 16)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    def record_smi(command, **kwargs):
        assert command == ["npu-smi", "info"]
        assert kwargs["timeout"] == 30
        return SimpleNamespace(returncode=1, stdout="", stderr="device query unavailable")

    monkeypatch.setattr(module.subprocess, "run", record_smi)
    report = module.collect_environment()
    encoded = json.dumps(report)
    assert "secret-not-for-logs" not in encoded
    assert "HF_TOKEN" not in encoded
    assert report["env"]["VIME_TEST_RESOURCE_CLASS"] == "npu-4"
    assert report["npu_device_count"] == 16  # report runtime visibility separately from the request
    assert report["env"]["ASCEND_VISIBLE_DEVICES"] == "12,13,14,15"
    assert report["npu_smi"]["returncode"] == 1
    assert "/sys/fs/cgroup/memory.max" in report["cgroup"]
    assert "/dev/shm" in report["filesystems"]
