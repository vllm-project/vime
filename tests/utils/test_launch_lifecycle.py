"""Launcher contracts: commands are recorded; no Ray or training jobs run."""

import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vime.utils.external_utils import command_utils
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0


@pytest.mark.parametrize("platform_name", ["cuda", "npu"])
@pytest.mark.parametrize("multinode", [False, True])
def test_convert_checkpoint_uses_main_command_on_all_platforms(monkeypatch, tmp_path, platform_name, multinode):
    execute = Mock()
    monkeypatch.setattr(command_utils, "exec_command", execute)
    monkeypatch.setattr(command_utils, "current_platform", lambda: PLATFORMS[platform_name])
    monkeypatch.setenv("SLURM_JOB_HOSTNAMES", "node0\nnode1")
    monkeypatch.setenv("SLURM_NODEID", "1")

    command_utils.convert_checkpoint(
        "model",
        "qwen3-4B",
        4,
        multinode=multinode,
        dir_dst=str(tmp_path),
        hf_checkpoint="/models/hf",
        extra_args=" --use-cpu-initialization ",
    )

    execute.assert_called_once()
    command = execute.call_args.args[0]
    assert "scripts/models/qwen3-4B.sh" in command
    assert "torchrun --nproc-per-node 4" in command
    assert "tools/convert_hf_to_torch_dist.py" in command
    assert f"--hf-checkpoint /models/hf --save {tmp_path}/model_torch_dist --use-cpu-initialization" in command
    assert ("--master-addr node0 --master-port 23456 --nnodes=2 --node-rank 1" in command) == multinode

    (tmp_path / "model_torch_dist").mkdir()
    execute.reset_mock()
    command_utils.convert_checkpoint("model", "qwen3-4B", 4, dir_dst=str(tmp_path))
    execute.assert_not_called()


@pytest.mark.parametrize("platform_name", ["cuda", "npu"])
@pytest.mark.parametrize("external_ray,live_head", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("submit", [False, True])
def test_execute_train_preserves_cluster_and_serving(monkeypatch, platform_name, external_ray, live_head, submit):
    events = []
    status = Mock(return_value=SimpleNamespace(returncode=0 if live_head else 1))
    monkeypatch.setattr(command_utils, "current_platform", lambda: PLATFORMS[platform_name])
    monkeypatch.setattr(command_utils, "exec_command", events.append)
    monkeypatch.setattr(command_utils.subprocess, "run", status)
    monkeypatch.setattr(command_utils, "check_has_nvlink", lambda: False)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.2")
    monkeypatch.setenv("VIME_SCRIPT_EXTERNAL_RAY", str(int(external_ray)))
    monkeypatch.setenv("VIME_SCRIPT_ENABLE_RAY_SUBMIT", str(int(submit)))

    command_utils.execute_train(
        "--num-rollout 2",
        num_gpus_per_node=4,
        megatron_model_type="qwen3-4B",
        before_ray_job_submit=lambda: events.append("before_submit"),
        extra_env_vars={"CUSTOM": "argument"},
        config=command_utils.ExecuteTrainConfig(extra_env_vars='{"CUSTOM": "config value"}'),
    )

    assert not any("pkill" in event or "ray stop" in event for event in events)
    starts = [event for event in events if "ray start" in event]
    submissions = [event for event in events if "ray job submit" in event]
    assert len(starts) == int(not external_ray and not live_head)
    assert len(submissions) == int(submit)
    assert events == starts + ["before_submit"] + submissions
    if external_ray:
        status.assert_not_called()
    else:
        status.assert_called_once_with(
            ["ray", "status"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
    if starts:
        assert "--node-ip-address 127.0.0.2" in starts[0]
        if platform_name == "npu":
            assert "--num-gpus 0" in starts[0]
            assert "--resources '{\"NPU\": 4}'" in starts[0]
            assert "export VIME_PLATFORM=npu" in starts[0]
            assert "export CUSTOM='config value'" in starts[0]
            assert "VLLM_ASCEND_ENABLE_NZ" not in starts[0]
        else:
            assert "--num-gpus 4" in starts[0]
            assert "--resources" not in starts[0]
    if submissions:
        command = submissions[0]
        assert "-- python3 train.py " in command
        assert "train_async.py" not in command
        assert "scripts/models/qwen3-4B.sh" in command
        assert "${MODEL_ARGS[@]}" in command
        assert "--num-rollout 2" in command
        runtime_arg = next(arg for arg in shlex.split(command) if arg.startswith("--runtime-env-json="))
        env = json.loads(runtime_arg.split("=", 1)[1])["env_vars"]
        assert env["CUSTOM"] == "config value"
        assert env["MASTER_ADDR"] == "127.0.0.2"
        if platform_name == "npu":
            assert env["VIME_PLATFORM"] == "npu"
            assert env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] == "1"
            assert "VLLM_ASCEND_ENABLE_NZ" not in env


@pytest.mark.parametrize("platform_name", ["cuda", "npu"])
def test_ray_status_timeout_does_not_start_or_submit(monkeypatch, platform_name):
    execute = Mock()
    callback = Mock()
    monkeypatch.setattr(command_utils, "current_platform", lambda: PLATFORMS[platform_name])
    monkeypatch.setattr(command_utils, "exec_command", execute)
    monkeypatch.setattr(
        command_utils.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired(["ray", "status"], 30))
    )
    monkeypatch.setenv("VIME_SCRIPT_EXTERNAL_RAY", "0")

    with pytest.raises(subprocess.TimeoutExpired):
        command_utils.execute_train("", 4, None, before_ray_job_submit=callback)

    execute.assert_not_called()
    callback.assert_not_called()


@pytest.mark.parametrize("external_ray,live_head", [("0", False), ("false", True), ("1", False), ("TRUE", False)])
def test_fully_async_npu_example_preserves_cluster(tmp_path, external_ray, live_head):
    # Only the shell's control flow runs: every process-management command is a
    # recording executable. No Python trainer or actual Ray CLI is invoked.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "commands"
    ray = bin_dir / "ray"
    ray.write_text(
        "#!/bin/bash\n"
        'printf "%s\\0" "$@" >> "$LAUNCH_TEST_LOG"\n'
        'printf "\\n" >> "$LAUNCH_TEST_LOG"\n'
        'if [[ "$1" == status ]]; then exit "$LAUNCH_TEST_STATUS"; fi\n'
        'if [[ "$1" == start || "$1 $2" == "job submit" ]]; then exit 0; fi\n'
        "exit 99\n"
    )
    ray.chmod(0o755)
    for name in ("pkill", "killall", "sleep", "python", "python3"):
        forbidden = bin_dir / name
        forbidden.write_text(f'#!/bin/bash\nprintf "forbidden:{name}\\0\\n" >> "$LAUNCH_TEST_LOG"\nexit 99\n')
        forbidden.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "VIME_SCRIPT_EXTERNAL_RAY": external_ray,
        "LAUNCH_TEST_LOG": str(log_path),
        "LAUNCH_TEST_STATUS": "0" if live_head else "1",
    }
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["bash", str(repo / "examples/fully_async/run-qwen3-4B-fully_async-npu.sh")],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        timeout=10,
    )
    calls = [line.split(b"\0")[:-1] for line in log_path.read_bytes().splitlines()]
    external = external_ray.lower() in ("1", "true")
    assert [call[0] for call in calls] == ([] if external else [b"status"]) + (
        [b"start"] if not external and not live_head else []
    ) + [b"job"]
    assert calls[-1][:2] == [b"job", b"submit"]
    entrypoint = calls[-1].index(b"--")
    assert calls[-1][entrypoint + 1 : entrypoint + 3] == [b"python3", b"train.py"]
    rollout = calls[-1].index(b"--rollout-function-path")
    assert calls[-1][rollout + 1] == b"vime.rollout.fully_async_rollout.generate_rollout_fully_async"
