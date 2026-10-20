"""CPU contracts for shared lifecycle/recovery tests; no hardware or Ray runs."""

import importlib.util
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0
TESTS = Path(__file__).resolve().parents[1]


def load_case(filename):
    spec = importlib.util.spec_from_file_location("lifecycle_case", TESTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(args, name):
    # Some recovery callers override the base topology or memory policy.
    return args[len(args) - 1 - args[::-1].index(name) + 1]


@pytest.fixture(params=["cuda", "npu"])
def platform_case(request, monkeypatch, tmp_path):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[request.param])
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path / "assets with spaces"))
    monkeypatch.setenv("VIME_TEST_ENABLE_EVAL", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.syspath_prepend(str(TESTS))
    monkeypatch.setattr(U, "exec_command", lambda command: None)
    return request.param


def assert_common(args, module, platform):
    assert option(args, "--hf-checkpoint") == str(module.MODEL_DIR)
    assert option(args, "--prompt-data").startswith(str(module.DATASET_DIR) + "/")
    assert "--vllm-enforce-eager" not in args
    if platform == "npu":
        assert "--use-flash-attn" in args and "--no-gradient-accumulation-fusion" in args
        assert option(args, "--log-probs-chunk-size") == "128"
    else:
        assert "--no-gradient-accumulation-fusion" not in args


@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_mixed_offload_keeps_frozen_group_and_config_lifetime(platform_case, fault_tolerance, monkeypatch):
    module = load_case(
        "test_vllm_config_mixed_offload_ft.py" if fault_tolerance else "test_vllm_config_mixed_offload.py"
    )
    configs = []

    def record(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert_common(args, module, platform_case)
        assert "--colocate" in args and "--ci-test" in args
        assert ("--use-fault-tolerance" in args) == fault_tolerance
        assert option(args, "--num-rollout") == ("3" if fault_tolerance else "2")
        assert "--eval-interval" not in args
        path = Path(option(args, "--vllm-config"))
        configs.append(path)
        content = yaml.safe_load(path.read_text())["vllm"]
        assert [(m["name"], m["update_weights"]) for m in content] == [("actor", True), ("ref", False)]
        assert [m["server_groups"][0]["num_gpus"] for m in content] == [4, 4]
        if platform_case == "npu":
            assert path.is_relative_to(module.TEST_ROOT / "tmp")
            assert option(args, "--vllm-max-num-seqs") == "16"

    monkeypatch.setattr(U, "execute_train", record)
    module.execute()
    assert configs and all(not path.exists() for path in configs)


def test_release_train_keeps_two_real_checkpoint_cycles(platform_case, monkeypatch):
    module = load_case("test_release_train.py")

    def record(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert_common(args, module, platform_case)
        assert "--release-train" in args and "--colocate" in args
        assert option(args, "--save-interval") == "1" and option(args, "--num-rollout") == "2"
        assert option(args, "--update-weight-mode") == "full" and option(args, "--update-weight-transport") == "disk"
        assert option(args, "--entropy-coef") == "0.01"
        assert option(args, "--rollout-max-response-len") == "512"
        assert option(args, "--ref-load") == (
            str(module.MODEL_DIR) if platform_case == "npu" else module.TORCH_DIST_CKPT
        )
        directory = Path(option(args, "--save"))
        directory.mkdir(parents=True)
        (directory / "latest_checkpointed_iteration.txt").write_text("1")

    monkeypatch.setattr(U, "execute_train", record)
    module.execute()


def test_debug_replay_keeps_straw_and_pt_without_starting_new_serving(platform_case, monkeypatch, tmp_path):
    module = load_case("test_qwen2.5_0.5B_debug_rollout_then_train.py")
    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kw: calls.append(shlex.split(kw["train_args"])))
    directory = str(tmp_path / "archive with spaces")
    module.execute_rollout_only(directory)
    module.execute_train_only(directory, "straw.json")
    module.execute_train_only(directory, "pt")
    assert len(calls) == 3
    for args in calls:
        assert_common(args, module, platform_case)
        assert option(args, "--num-rollout") == "2" and option(args, "--rollout-max-response-len") == "256"
    assert "--debug-rollout-only" in calls[0]
    assert option(calls[0], "--save-debug-rollout-data").startswith(directory + "/")
    for args, extension in zip(calls[1:], ["straw.json", "pt"], strict=True):
        assert option(args, "--load-debug-rollout-data") == directory + "/rollout_{rollout_id}." + extension
        assert "--debug-rollout-only" not in args and "--ci-test" in args
        assert "--vllm-gpu-memory-utilization" not in args


class LaunchRecorded(Exception):
    pass


def test_straw_fork_uses_device_policy_with_original_disaggregated_layout(platform_case, monkeypatch):
    module = load_case("test_straw_checkpoint_fork.py")

    def record(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert_common(args, module, platform_case)
        assert option(args, "--actor-num-gpus-per-node") == "1" and option(args, "--rollout-num-gpus") == "3"
        assert option(args, "--num-rollout") == "3" and option(args, "--save-interval") == "1"
        assert option(args, "--entropy-coef") == "0.01" and "--rollout-queue-online-gc" in args
        assert Path(option(args, "--rollout-data-dir")).parent.is_dir()
        raise LaunchRecorded

    monkeypatch.setattr(U, "execute_train", record)
    with pytest.raises(LaunchRecorded):
        module.execute()


@pytest.mark.parametrize(
    ("save", "load", "async_save"),
    [("gpu", "gpu", False), ("gpu", "cpu", False), ("cpu", "cpu", False), ("cpu", "gpu", False), ("cpu", "cpu", True)],
)
def test_checkpoint_variants_restore_optimizer_and_train_after_load(
    platform_case, monkeypatch, tmp_path, save, load, async_save
):
    module = load_case("test_qwen3_4B_ckpt.py")
    hook = load_case("checkpoint_test_helpers.py")
    monkeypatch.setattr(hook.dist, "get_rank", lambda: 0)
    calls = []

    def record(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert_common(args, module, platform_case)
        phase = kwargs["extra_env_vars"]["VIME_CKPT_TEST_PHASE"]
        calls.append(phase)
        expected_optimizer = save if phase == "save" else load
        assert ("--optimizer-cpu-offload" in args) == (expected_optimizer == "cpu")
        assert "--use-precision-aware-optimizer" in args
        assert ("--async-save" in args) == (async_save and phase == "save")
        assert "--no-save-optim" not in args and "--no-load-optim" not in args
        assert option(args, "--tensor-model-parallel-size") == option(args, "--context-parallel-size") == "2"
        assert option(args, "--ref-load") == (
            str(module.MODEL_DIR) if platform_case == "npu" else str(module.MODEL_DIR) + "_torch_dist"
        )
        if phase == "load":
            assert option(args, "--ckpt-step") == "1" and option(args, "--num-rollout") == "3"
            assert "--use-checkpoint-opt-param-scheduler" in args
        else:
            assert option(args, "--num-rollout") == "2"
            queue = Path(option(args, "--save")) / "rollout"
            queue.mkdir(parents=True)
            for name in ("queue_state_1.json", "builder_state_1.json"):
                (queue / name).write_text("{}")
        for key, value in kwargs["extra_env_vars"].items():
            monkeypatch.setenv(key, value)
        for rollout in (0, 1) if phase == "save" else (2,):
            hook.before_train_step(
                SimpleNamespace(
                    global_batch_size=16,
                    no_load_optim=phase == "save",
                    finetune=phase == "save",
                    optimizer_cpu_offload=expected_optimizer == "cpu",
                ),
                rollout,
                0,
                [],
                None,
                SimpleNamespace(num_steps=rollout * 16),
            )
            path = Path(option(args, "--ci-save-grad-norm").format(rollout_id=rollout, step_id=0))
            torch.save(torch.tensor(0.5), path)

    monkeypatch.setattr(U, "execute_train", record)
    module.run_roundtrip(
        SimpleNamespace(
            async_save=async_save,
            save_optimizer=save,
            load_optimizer=load,
            checkpoint_dir=str(tmp_path / "checkpoint"),
        )
    )
    assert calls == ["save", "load"]


@pytest.mark.parametrize("broken", ["missing", "nan", "scheduler", "optimizer"])
def test_checkpoint_verification_rejects_incomplete_restore(monkeypatch, tmp_path, broken):
    module = load_case("test_qwen3_4B_ckpt.py")
    directory = tmp_path / "checks/load"
    directory.mkdir(parents=True)
    if broken != "missing":
        torch.save(torch.tensor(float("nan") if broken == "nan" else 0.5), directory / "grad_2_0.pt")
    (directory / "before_2.json").write_text(
        json.dumps(
            dict(
                rollout_id=2,
                scheduler_num_steps=0 if broken == "scheduler" else 32,
                no_load_optim=broken == "optimizer",
                finetune=False,
            )
        )
    )
    with pytest.raises(AssertionError):
        module.verify_phase(tmp_path, "load", (2,))


def test_checkpoint_does_not_delete_an_existing_user_directory(monkeypatch, tmp_path):
    module = load_case("test_qwen3_4B_ckpt.py")
    sentinel = tmp_path / "do-not-delete"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        module.prepare(str(tmp_path))
    assert sentinel.read_text() == "preserve"


def test_recovery_head_advertises_selected_resource_and_vendor_environment(platform_case, monkeypatch, tmp_path):
    module = load_case("test_qwen2.5_0.5B_training_recovery.py")
    starts = []
    monkeypatch.setenv("VIME_SCRIPT_EXTERNAL_RAY", "0")
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: starts.append((args, kw)))

    def stop_before_jobs(*args, **kwargs):
        raise LaunchRecorded

    monkeypatch.setattr(module, "JobSubmissionClient", stop_before_jobs)
    with pytest.raises(LaunchRecorded):
        module.execute(directory=tmp_path)
    assert len(starts) == 1
    command, kwargs = starts[0]
    assert option(command, "--num-gpus") == ("0" if platform_case == "npu" else "4")
    env = kwargs["env"]
    assert env["VIME_PLATFORM"] == platform_case
    assert env["PYTHONPATH"].split(":")[0] == str(TESTS)
    if platform_case == "npu":
        assert json.loads(option(command, "--resources")) == {"NPU": 4}
        assert "/root/MegatronAdaptor" in env["PYTHONPATH"] and "ascend-toolkit" in env["LD_LIBRARY_PATH"]
        assert "HCCL_HOST_SOCKET_PORT_RANGE" in env


@pytest.mark.parametrize("mode", ["server", "actor"])
def test_health_recovery_keeps_crash_mode_and_router_evidence(platform_case, monkeypatch, tmp_path, mode):
    module = load_case("test_qwen2.5_0.5B_rollout_health.py")

    def record(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert_common(args, module, platform_case)
        assert option(args, "--num-rollout") == "2"
        assert (
            option(args, "--custom-megatron-before-train-step-hook-path")
            == "rollout_health_test_helpers.before_train_step"
        )
        assert option(args, "--rollout-function-path") == "rollout_health_test_helpers.generate_rollout"
        assert kwargs["extra_env_vars"]["VIME_HEALTH_CRASH_MODE"] == mode
        directory = Path(kwargs["extra_env_vars"]["VIME_HEALTH_TEST_DIR"])
        (directory / "crash.json").write_text(json.dumps({"drain_seconds": 1.0}))
        for rollout, live in [(0, 1), (1, 2)]:
            (directory / f"trained_{rollout}.json").write_text(json.dumps({"live_engines": live}))
        (directory / "checkpoint/iter_0000001").mkdir(parents=True)

    def no_retained_actor(*args, **kwargs):
        raise ValueError("none")

    monkeypatch.setattr(U, "execute_train", record)
    monkeypatch.setattr(module.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(module.ray, "shutdown", lambda: None)
    monkeypatch.setattr(module.ray, "get_actor", no_retained_actor)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    module.execute(mode, directory=tmp_path)


@pytest.mark.parametrize("tp", [1, 2])
def test_recovery_keeps_failure_hook_and_replay_contract(platform_case, tp, tmp_path):
    module = load_case("test_qwen2.5_0.5B_training_recovery.py")
    args = shlex.split(
        module.train_args(
            tmp_path, "straw", tp, str(module.MODEL_DIR), str(module.DATASET_DIR / "gsm8k/train.parquet"), 3, False
        )
    )
    assert_common(args, module, platform_case)
    assert option(args, "--tensor-model-parallel-size") == str(tp)
    assert option(args, "--num-rollout") == "3" and option(args, "--save-interval") == "3"
    assert (
        option(args, "--custom-megatron-before-train-step-hook-path")
        == "training_recovery_test_helpers.before_train_step"
    )
    assert "--rollout-queue-online-gc" in args and "--use-fault-tolerance" not in args


@pytest.mark.parametrize("device", ["cuda:2", "npu:3"])
def test_real_oom_injection_targets_the_selected_allocator(monkeypatch, device):
    module = load_case("training_recovery_test_helpers.py")
    monkeypatch.setattr(module.accelerator, "device", lambda: device)
    monkeypatch.setattr(module.accelerator, "get_device_properties", lambda: SimpleNamespace(total_memory=64))
    calls = []

    def allocate(size, **kwargs):
        calls.append((size, kwargs))
        raise torch.OutOfMemoryError("allocator exhausted")

    monkeypatch.setattr(module.torch, "empty", allocate)
    with pytest.raises(torch.OutOfMemoryError):
        module.inject_device_oom()
    assert calls == [(128, {"dtype": torch.uint8, "device": device})]


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        ("CUDA out of memory", True),
        ("NPU out of memory", True),
        ("torch_npu._C.OutOfMemoryError: failure", True),
        ("HCCL connection timeout", False),
        ("RuntimeError: unsupported layout", False),
    ],
)
def test_oom_recognition_rejects_unrelated_initialization_errors(log, expected):
    module = load_case("training_recovery_test_helpers.py")
    assert module.is_device_oom(log) is expected
