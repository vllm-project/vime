"""Check CI paths/topology without downloading weights or starting Ray."""

import importlib
import runpy
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_sparse_npu_case_uses_hf_home_and_noncolocated_topology(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/tmp/hf cache")
    commands = Mock()
    launch = Mock()
    importlib.import_module("vime.utils.external_utils")
    monkeypatch.setitem(
        sys.modules,
        "vime.utils.external_utils.command_utils",
        SimpleNamespace(exec_command=commands, execute_train=launch),
    )
    case = runpy.run_path(str(REPO_ROOT / "tests/test_qwen3_30B_A3B_sparse_hccl_npu.py"))
    case["prepare"]()
    case["execute"]()
    assert case["MODEL_DIR"] == "/tmp/hf cache/models/Qwen3-30B-A3B"
    assert case["DATASET_DIR"] == "/tmp/hf cache/datasets/dapo-math-17k"
    assert shlex.split(commands.call_args_list[1].args[0])[-1] == case["MODEL_DIR"]
    assert shlex.split(commands.call_args_list[2].args[0])[-1] == case["DATASET_DIR"]
    bridge_install = shlex.split(commands.call_args_list[3].args[0])
    assert "--no-deps" in bridge_install
    assert bridge_install[-1].endswith("@" + case["BRIDGE_COMMIT"])
    config = launch.call_args.kwargs
    args = shlex.split(config["train_args"])
    for flag, value in {
        "--hf-checkpoint": case["MODEL_DIR"],
        "--prompt-data": f"{case['DATASET_DIR']}/dapo-math-17k.jsonl",
        "--tensor-model-parallel-size": "2",
        "--pipeline-model-parallel-size": "2",
        "--expert-model-parallel-size": "2",
        "--expert-tensor-parallel-size": "1",
        "--actor-num-gpus-per-node": "4",
        "--rollout-num-gpus": "4",
        "--rollout-num-gpus-per-engine": "4",
        "--vllm-pipeline-parallel-size": "2",
        "--update-weight-mode": "delta",
        "--update-weight-transport": "sparse_hccl",
        "--update-weight-delta-verify-every": "1",
        "--num-rollout": "3",
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--colocate" not in args
    assert "--vllm-enable-expert-parallel" in args
    assert config["num_gpus_per_node"] == 8
    assert config["extra_env_vars"]["VIME_SPARSE_HCCL_SNAPSHOT_DEVICE"] == "cpu"


@pytest.mark.unit
def test_sparse_npu_suite_requests_eight_devices_without_visibility_override(monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "smk")
    suites = runpy.run_path(str(REPO_ROOT / ".buildkite/npu_suites.py"))
    entries = [entry for entry in suites["SUITES"]["smk"] if entry[0] == "test_qwen3_30B_A3B_sparse_hccl_npu.py"]
    assert len(entries) == 1
    assert entries[0][3] == {}
    step = suites["npu_step"]("smk", *entries[0])
    assert step["agents"]["resource_class"] == "npu-8"
    assert step["env"]["HF_HOME"] == "/root/.cache/huggingface"
    assert "python tests/test_qwen3_30B_A3B_sparse_hccl_npu.py" in step["command"]
