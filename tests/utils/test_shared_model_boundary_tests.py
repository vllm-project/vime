"""CPU launch contracts for batch-four MoE, R3, MTP and streaming cases."""

import importlib.util
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0
TESTS = Path(__file__).resolve().parents[1]
CASES = {
    "test_qwen3_30B_A3B_r3.py": (4, 2, 8, 8192, 2, 16384),
    "test_moonlight_16B_A3B_non_colocate_nccl.py": (1, 1, 4, 512, 2, 4096),
    "test_moonlight_16B_A3B.py": (2, 2, 8, 4096, 2, 8192),
    "test_moonlight_16B_A3B_r3.py": (2, 2, 8, 4096, 2, 8192),
    "test_quick_start_glm4_9B.py": (2, 2, 1, 8192, 2, 4608),
    "test_mimo_7B_mtp_only_grad.py": (2, 1, 1, 128, 1, 4096),
    "test_qwen3_4B_streaming_partial_rollout.py": (2, 2, 1, 4096, 2, 8192),
}


def load_case(filename):
    spec = importlib.util.spec_from_file_location("model_boundary_case", TESTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(args, flag):
    return args[max(i for i, arg in enumerate(args) if arg == flag) + 1]


@pytest.fixture(params=["cuda", "npu"])
def platform(request, monkeypatch, tmp_path):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[request.param])
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path / "assets with spaces"))
    monkeypatch.setenv("VIME_TEST_ENABLE_EVAL", "0")
    monkeypatch.setenv("VIME_TEST_USE_DEEPEP", "0")
    monkeypatch.setenv("VIME_TEST_USE_FP8_ROLLOUT", "0")
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return request.param


@pytest.mark.parametrize("filename", CASES)
def test_model_cases_preserve_topology_and_feature_checks(platform, filename, monkeypatch):
    module = load_case(filename)
    tp, cp, ep, response, rounds, gpu_tokens = CASES[filename]
    calls = []
    scratch = []

    def record(**kwargs):
        args = shlex.split(kwargs["train_args"])
        calls.append(args)
        assert kwargs["num_gpus_per_node"] == 8
        assert option(args, "--hf-checkpoint") == str(module.MODEL_DIR)
        assert option(args, "--prompt-data") == str(module.DATASET_DIR / "dapo-math-17k/dapo-math-17k.jsonl")
        assert option(args, "--tensor-model-parallel-size") == str(tp)
        assert option(args, "--context-parallel-size") == str(cp)
        # Dense scripts inherit EP1 from Megatron rather than explicitly setting it.
        assert (
            option(args, "--expert-model-parallel-size") if "--expert-model-parallel-size" in args else "1"
        ) == str(ep)
        assert option(args, "--rollout-max-response-len") == str(response)
        assert option(args, "--num-rollout") == str(rounds)
        assert "--ci-test" in args and "--eval-prompt-data" not in args
        r3 = filename.endswith("_r3.py")
        assert ("--use-rollout-routing-replay" in args) == r3
        if "--rollout-data-dir" in args:
            path = Path(option(args, "--rollout-data-dir"))
            assert path.is_dir()
            if platform == "npu":
                assert path.is_relative_to(module.TEST_ROOT / "tmp")
            scratch.append(path)
        eager = filename in ("test_mimo_7B_mtp_only_grad.py", "test_moonlight_16B_A3B_non_colocate_nccl.py")
        assert ("--vllm-enforce-eager" in args) == eager
        if platform == "npu":
            assert option(args, "--ref-load") == str(module.MODEL_DIR)
            assert option(args, "--max-tokens-per-gpu") == "2048"
            assert option(args, "--log-probs-chunk-size") == "128"
            assert option(args, "--vllm-gpu-memory-utilization") == "0.5"
            context = 4096 if "non_colocate" in filename else 8192 if "moonlight" in filename else 32768
            assert option(args, "--vllm-max-model-len") == str(context)
            assert option(args, "--vllm-max-num-seqs") == ("8" if "non_colocate" in filename else "16")
            assert "--no-gradient-accumulation-fusion" in args
            assert "--optimizer-cpu-offload" in args and "--use-precision-aware-optimizer" in args
        else:
            assert option(args, "--max-tokens-per-gpu") == str(gpu_tokens)
            assert "--log-probs-chunk-size" not in args and "--no-gradient-accumulation-fusion" not in args
            assert ("--optimizer-cpu-offload" in args) == ("qwen3_30B" in filename or "non_colocate" in filename)
        if "non_colocate" in filename:
            assert "--colocate" not in args
            assert option(args, "--actor-num-gpus-per-node") == option(args, "--rollout-num-gpus") == "4"
            assert option(args, "--vllm-data-parallel-size") == "4" and "--vllm-enable-expert-parallel" in args
            assert option(args, "--vllm-all2all-backend") == "naive"
        if "streaming" in filename:
            assert (
                option(args, "--custom-generate-function-path")
                == "vime.rollout.vllm_streaming_rollout.generate_streaming"
            )
            assert option(args, "--over-sampling-batch-size") == "8"
            assert "--partial-rollout" in args and "--mask-offpolicy-in-partial-rollout" in args
        if "mimo" in filename:
            assert "--enable-mtp-training" in args and option(args, "--mtp-num-layers") == "1"
            assert option(args, "--mtp-loss-scaling-factor") == "0.2"
            assert json.loads(option(args, "--vllm-speculative-config")) == {
                "method": "mtp",
                "num_speculative_tokens": 3,
            }
            assert "--ci-disable-kl-checker" in args

    monkeypatch.setattr(U, "execute_train", record)
    module.execute()
    assert len(calls) == 1 and all(not path.exists() for path in scratch)


@pytest.mark.parametrize("filename", [*CASES, "test_qwen3_30B_A3B_training_recovery.py"])
def test_assets_use_the_same_root_and_skip_unused_downloads(platform, filename, monkeypatch):
    module = load_case(filename)
    commands, datasets, conversions = [], [], []
    monkeypatch.setattr(U, "exec_command", commands.append)
    monkeypatch.setattr(U, "hf_download_dataset", lambda name, **kw: datasets.append((name, kw)))
    monkeypatch.setattr(U, "convert_checkpoint", lambda **kw: conversions.append(kw))
    module.prepare()
    download = next(shlex.split(command) for command in commands if command.startswith("hf download"))
    assert option(download, "--local-dir") == str(module.MODEL_DIR)
    assert len(download[2].split("/")) == 2
    assert datasets and all(kw["dir_dst"] == module.DATASET_DIR for _, kw in datasets)
    assert all("aime" not in name for name, _ in datasets)
    assert not any("-FP8" in command for command in commands)
    if conversions:
        assert conversions[0]["hf_checkpoint"] == str(module.MODEL_DIR)


def test_moe_recovery_keeps_real_failure_reshard_routes_and_stateless_optimizer(platform, monkeypatch, tmp_path):
    module = load_case("test_qwen3_30B_A3B_training_recovery.py")
    calls = []
    monkeypatch.setattr(module.runpy, "run_path", lambda path: {"execute": lambda **kw: calls.append(kw)})
    module.execute(keep_artifacts=True)
    call = calls[0]
    assert call["model_path"] == str(module.MODEL_DIR)
    assert call["dataset_path"] == str(module.DATASET_DIR / "gsm8k/train.parquet")
    assert call["train_gpus"] == call["rollout_gpus"] == 4
    assert call["failure_rollout"] == call["save_interval"] == 1
    assert call["kill_manager"] and call["manager_crash_phase"] == "training" and not call["fault_tolerance"]
    assert not call["pd"] and call["keep_artifacts"]
    assert call["job_timeout_seconds"] == (10800 if platform == "npu" else 3600)
    args = shlex.split(call["extra_args"])
    for flag in (
        "--use-stateless-adam",
        "--no-save-optim",
        "--use-rollout-routing-replay",
        "--optimizer-cpu-offload",
        "--overlap-grad-reduce",
    ):
        assert flag in args
    assert option(args, "--expert-model-parallel-size") == "4"
    assert option(args, "--rollout-num-gpus-per-engine") == "2"
    assert option(args, "--moe-token-dispatcher-type") == "alltoall"
    assert json.loads(option(args, "--train-env-vars"))["OMP_NUM_THREADS"] == "8"
    module.execute(directory=tmp_path, job_timeout_seconds=1200)
    assert calls[-1]["job_timeout_seconds"] == 1200


@pytest.mark.parametrize("deepep,fp8", [(True, False), (False, True), (True, True)])
def test_npu_r3_rejects_gpu_specific_variants_before_download(monkeypatch, deepep, fp8):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS["npu"])
    monkeypatch.setenv("VIME_TEST_USE_DEEPEP", str(int(deepep)))
    monkeypatch.setenv("VIME_TEST_USE_FP8_ROLLOUT", str(int(fp8)))
    module = load_case("test_qwen3_30B_A3B_r3.py")
    monkeypatch.setattr(U, "exec_command", lambda command: pytest.fail("must fail before download"))
    with pytest.raises(ValueError, match="separate adaptation"):
        module.prepare()
    with pytest.raises(ValueError, match="separate adaptation"):
        module.execute()


@pytest.mark.parametrize("deepep,fp8", [(True, False), (False, True), (True, True)])
def test_gpu_r3_preserves_original_deepep_and_fp8_choices(monkeypatch, tmp_path, deepep, fp8):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS["cuda"])
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("VIME_TEST_ENABLE_EVAL", "0")
    monkeypatch.setenv("VIME_TEST_USE_DEEPEP", str(int(deepep)))
    monkeypatch.setenv("VIME_TEST_USE_FP8_ROLLOUT", str(int(fp8)))
    module = load_case("test_qwen3_30B_A3B_r3.py")
    commands, calls = [], []
    monkeypatch.setattr(U, "exec_command", commands.append)
    monkeypatch.setattr(U, "hf_download_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(U, "convert_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(U, "execute_train", lambda **kw: calls.append(shlex.split(kw["train_args"])))
    module.prepare()
    module.execute()
    args = calls[0]
    assert any("-FP8" in command for command in commands) is fp8
    assert option(args, "--hf-checkpoint") == str(module.MODEL_DIR) + ("-FP8" if fp8 else "")
    assert ("--moe-enable-deepep" in args) is deepep
    assert option(args, "--moe-token-dispatcher-type") == ("flex" if deepep else "alltoall")
    if deepep:
        assert option(args, "--vllm-all2all-backend") == "deepep_high_throughput"
    assert "--use-rollout-routing-replay" in args


def test_recovery_deadline_keeps_tail_and_stops_only_its_job(monkeypatch, tmp_path):
    module = load_case("test_qwen2.5_0.5B_training_recovery.py")
    times = iter([0, 11])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    stopped = []
    client = SimpleNamespace(get_job_logs=lambda job: "fixture failure tail", stop_job=stopped.append)
    log = tmp_path / "failed.log"
    with pytest.raises(TimeoutError, match="within 10s"):
        module.wait_job(client, "owned-job", log, timeout_seconds=10)
    assert stopped == ["owned-job"] and log.read_text() == "fixture failure tail"
    with pytest.raises(ValueError, match="positive"):
        module.execute(job_timeout_seconds=0)
