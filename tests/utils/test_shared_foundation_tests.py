"""Batch-five launch, process ownership and numerical-oracle checks on CPU."""

import importlib.util
import os
import shlex
import sys
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vime.platforms import get_platform
from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0
TESTS = Path(__file__).resolve().parents[1]


def load_case(monkeypatch, filename):
    monkeypatch.syspath_prepend(str(TESTS))
    name = filename.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, TESTS / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def option(args, flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]


@pytest.fixture(params=["cuda", "npu"])
def platform(request, monkeypatch, tmp_path):
    name = request.param
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[name])
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path / "assets with spaces"))
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "0")
    monkeypatch.setenv("VIME_TEST_ENABLE_EVAL", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return get_platform(name)


def test_basic_training_keeps_gpu_default_and_explicit_npu_transport(platform, monkeypatch):
    module = load_case(monkeypatch, "test_qwen3.5_0.8B_gsm8k_short.py")
    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    module.execute(**({"transport": "object-store"} if platform.is_npu else {}))
    args = shlex.split(calls[0]["train_args"])
    assert calls[0]["num_gpus_per_node"] == 4
    assert option(args, "--hf-checkpoint") == str(module.MODEL_DIR)
    assert option(args, "--ref-load") == str(module.MODEL_DIR if platform.is_npu else module.TORCH_DIST_CKPT)
    assert option(args, "--prompt-data") == str(module.DATASET_DIR / "gsm8k/train.parquet")
    assert option(args, "--rollout-data-transport") == ("object-store" if platform.is_npu else "nixl")
    assert option(args, "--num-rollout") == "3"
    assert option(args, "--rollout-max-response-len") == "1024"
    assert option(args, "--rollout-top-p") == "0.95"
    assert option(args, "--loss-mask-type") == "qwen3_5"
    assert "--use-fault-tolerance" in args and "--ci-test" in args
    assert "--dynamic-sampling-filter-path" in args and "--colocate" in args
    assert "--eval-prompt-data" not in args
    if platform.is_npu:
        assert option(args, "--vllm-max-model-len") == "8192"
        assert "--no-gradient-accumulation-fusion" in args
    with pytest.raises(ValueError, match="Unsupported transport"):
        module.execute("invalid")


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_opd_separates_teacher_and_restores_visibility_before_scratch_cleanup(platform, monkeypatch, keep, fail):
    module = load_case(monkeypatch, "test_qwen2.5_0.5B_opd_vllm.py")
    monkeypatch.setattr(module, "current_platform", lambda: platform)
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", str(int(keep)))
    visible = "7,6,5,4,3,2,1,0"
    monkeypatch.setenv(platform.ray.visible_devices_env, visible)
    scratch = []
    stopped = []
    process = object()

    def teacher(device, log):
        assert device == "3" and os.environ[platform.ray.visible_devices_env] == "7,6,5,4"
        log.write_text("fixture teacher ready")
        scratch.append(log.parent)
        return process

    def stop(owned):
        assert owned is process and scratch[0].exists()
        stopped.append(owned)

    def train(**kwargs):
        assert kwargs["num_gpus_per_node"] == 4
        args = shlex.split(kwargs["train_args"])
        assert "--use-opd" in args and option(args, "--opd-type") == "vllm"
        assert option(args, "--opd-kl-coef") == "1.0"
        assert option(args, "--num-rollout") == "2"
        assert option(args, "--rm-url").endswith("/inference/v1/generate")
        assert option(args, "--hf-checkpoint") == str(module.MODEL_DIR)
        kwargs["before_ray_job_submit"]()
        if fail:
            raise RuntimeError("fixture training failure")

    monkeypatch.setattr(module, "_launch_teacher_server", teacher)
    monkeypatch.setattr(module, "_stop_teacher", stop)
    monkeypatch.setattr(U, "execute_train", train)
    if fail:
        with pytest.raises(RuntimeError, match="fixture training failure"):
            module.execute()
    else:
        module.execute()
    assert stopped == [process]
    assert os.environ[platform.ray.visible_devices_env] == visible
    assert scratch[0].exists() is keep


@pytest.mark.parametrize("startup_failure", [False, True])
def test_teacher_launch_uses_platform_environment_and_cleans_failed_startup(
    platform, monkeypatch, tmp_path, startup_failure, capsys
):
    module = load_case(monkeypatch, "test_qwen2.5_0.5B_opd_vllm.py")
    monkeypatch.setattr(module, "current_platform", lambda: platform)
    monkeypatch.setattr(module, "is_port_available", lambda port: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    process = Mock(pid=123, returncode=9)
    process.poll.return_value = 9 if startup_failure else None
    launches = []

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        kwargs["stdout"].write("fixture teacher startup log")
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    response = Mock()
    response.__enter__ = Mock(return_value=SimpleNamespace(status=200))
    response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: response)
    stopped = []
    monkeypatch.setattr(module, "kill_process_tree", stopped.append)
    log = tmp_path / "teacher.log"
    if startup_failure:
        with pytest.raises(RuntimeError, match="exited"):
            module._launch_teacher_server("6", log)
        assert stopped == [123]
        process.wait.assert_called_once_with(timeout=30)
        assert "fixture teacher startup log" in capsys.readouterr().out
    else:
        assert module._launch_teacher_server("6", log) is process
        assert not stopped
    command, kwargs = launches[0]
    env = kwargs["env"]
    assert env[platform.ray.visible_devices_env] == "6"
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert option(command, "--model") == str(module.MODEL_DIR)
    if platform.is_npu:
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert option(command, "--max-model-len") == "8192"


@pytest.mark.parametrize("failure", [False, True])
def test_vlm_preserves_protocol_assertions_and_stops_services_before_removing_logs(
    platform, monkeypatch, failure, capsys
):
    module = load_case(monkeypatch, "test_geo3k_vlm_multi_turn_e2e.py")
    monkeypatch.setattr(module, "current_platform", lambda: platform)
    monkeypatch.setattr(module, "is_port_available", lambda port: True)
    monkeypatch.setenv(platform.ray.visible_devices_env, "3,2,1,0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    starts, stops = [], []
    directories = []

    def start(command, url, *, log_path, env=None):
        log_path.write_text("fixture service log")
        directories.append(log_path.parent)
        starts.append((command, env))
        return log_path.name

    def stop(process):
        assert directories[0].exists()
        stops.append(process)

    monkeypatch.setattr(module, "_start_process", start)
    monkeypatch.setattr(module, "_stop_process", stop)
    with pytest.raises(RuntimeError) if failure else nullcontext():
        with module._vllm_services():
            if failure:
                raise RuntimeError("fixture protocol failure")
    assert stops == ["router.log", "worker.log"]
    assert not directories[0].exists()
    command, env = starts[0]
    assert env[platform.ray.visible_devices_env] == "3"
    assert ("VLLM_BATCH_INVARIANT" in env) is not platform.is_npu
    assert option(command, "--tensor-parallel-size") == "1" and "--enforce-eager" in command
    assert str(module.MODEL_PATH) in command
    if failure:
        assert "fixture service log" in capsys.readouterr().out

    events = [
        module.RequestEvent(module.RENDER_ENDPOINT, (), (), False),
        module.RequestEvent(module.GENERATE_ENDPOINT, (1, 2), (3,), True),
        module.RequestEvent(module.GENERATE_ENDPOINT, (1, 2, 3, 4), (5,), True),
    ]
    sample = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5], response_length=3, loss_mask=[1, 0, 1], rollout_log_probs=[0, 0, 0]
    )
    module._validate_result(sample, events)
    sample.tokens[2] = 99
    with pytest.raises(AssertionError):
        module._validate_result(sample, events)


@pytest.mark.parametrize("scenario", product((-1, 2), (False, True), (0, 1, 2)))
def test_npu_mathematical_oracle_checks_mask_entropy_and_gradients_on_cpu(monkeypatch, scenario):
    module = load_case(monkeypatch, "test_ppo_logprob_entropy_npu.py")
    chunk, mask, entropy = scenario
    module.assert_reference_case(torch.device("cpu"), None, (chunk, mask, entropy > 0, entropy == 2))


def _reference_tp2_worker(rank, port):
    from datetime import timedelta

    import torch.distributed as dist

    sys.path.insert(0, str(TESTS))
    from test_ppo_logprob_entropy_npu import PARITY_SCENARIOS, assert_reference_case

    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for scenario in PARITY_SCENARIOS:
            assert_reference_case(torch.device("cpu"), dist.group.WORLD, scenario)
    finally:
        dist.destroy_process_group()


def test_numerical_oracle_covers_both_vocabulary_shards_with_collectives(monkeypatch):
    module = load_case(monkeypatch, "test_ppo_logprob_entropy_npu.py")
    torch.multiprocessing.spawn(_reference_tp2_worker, args=(module._free_port(),), nprocs=2, join=True)
