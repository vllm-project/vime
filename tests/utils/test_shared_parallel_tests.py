"""Exercise batch-two launch contracts and dump checks without starting hardware."""

import importlib.util
import shlex
from pathlib import Path

import pytest
import torch
import yaml

from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0
TESTS = Path(__file__).resolve().parents[1]
CASES = (
    "test_qwen2.5_0.5B_vllm_config.py",
    "test_qwen2.5_0.5B_vllm_config_distributed.py",
    "test_qwen2_5_0_5B_non_colocate_pp.py",
    "test_qwen3_5_0_8B_top_p_cp2.py",
    "test_qwen3_0.6B_parallel_check.py",
    "test_qwen2.5_0.5B_debug_train_dump_e2e.py",
    "test_qwen3_4B_ppo.py",
    "test_qwen3_4B_ppo_train_critic_only.py",
    "test_qwen3_4B_ppo_disaggregate.py",
)


def load_case(filename):
    spec = importlib.util.spec_from_file_location("shared_parallel_case", TESTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dump(directory, *, broken=None):
    path = Path(directory)
    (path / "rollout_data").mkdir(parents=True, exist_ok=True)
    (path / "train_data").mkdir(parents=True, exist_ok=True)
    torch.save({"samples": [{"rollout_log_probs": [-0.5, -0.25]}]}, path / "rollout_data/0.pt")
    sample = {"rollout_position": 0, "rollout_log_probs": torch.tensor([-0.5, -0.25])}
    if broken == "values":
        sample["rollout_log_probs"] = torch.tensor([-1.0, -0.25])
    elif broken == "position":
        sample["rollout_position"] = None
    elif broken == "no_comparison":
        sample.pop("rollout_log_probs")
    torch.save({"format_version": 2, "samples": [sample]}, path / "train_data/0.pt")


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("filename", CASES)
def test_parallel_cases_preserve_layouts_and_data(monkeypatch, tmp_path, platform, filename):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[platform])
    root = tmp_path / "assets with spaces"
    monkeypatch.setenv("VIME_TEST_ROOT", str(root))
    monkeypatch.setenv("VIME_TEST_ENABLE_EVAL", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    commands = []
    monkeypatch.setattr(U, "exec_command", commands.append)
    module = load_case(filename)
    calls = []
    configs = []

    def record_train(**kwargs):
        args = shlex.split(kwargs["train_args"])
        calls.append(args)

        def option(flag):
            return args[args.index(flag) + 1]

        assert option("--hf-checkpoint") == str(root / "models" / module.MODEL_NAME)
        assert option("--prompt-data").startswith(str(root / "datasets") + "/")
        assert "--ci-test" in args
        assert "--eval-interval" not in args
        assert "--vllm-enforce-eager" not in args
        if platform == "npu":
            assert option("--ref-load") == option("--hf-checkpoint")
            assert "--use-flash-attn" in args and "--no-gradient-accumulation-fusion" in args
            assert option("--log-probs-chunk-size") == "128"
            assert option("--vllm-max-num-seqs") == "16"
            assert option("--vllm-max-cudagraph-capture-size") == "8"
        else:
            assert "--log-probs-chunk-size" not in args
            assert "--no-gradient-accumulation-fusion" not in args

        for flag in ("--vllm-config", "--megatron-config-path"):
            if flag in args:
                config = Path(option(flag))
                configs.append(config)
                content = yaml.safe_load(config.read_text())
                if platform == "npu":
                    assert config.is_relative_to(root / "tmp")
                if flag == "--vllm-config":
                    groups = content["vllm"][0]["server_groups"]
                    distributed = "distributed" in filename
                    assert groups == [
                        {"worker_type": "regular", "num_gpus": 2 if distributed else 4, "num_gpus_per_engine": 2},
                        {"worker_type": "regular", "num_gpus": 1 if distributed else 2, "num_gpus_per_engine": 1},
                        {"worker_type": "placeholder", "num_gpus": 1 if distributed else 2},
                    ]
                    assert option("--actor-num-gpus-per-node") == ("4" if distributed else "8")
                    assert ("--colocate" in args) != distributed
                    assert option("--num-rollout") == "2"
                    if distributed:
                        assert option("--rollout-data-transport") == "straw"
                else:
                    assert {v["role"]: float(v["overrides"]["lr"]) for v in content["megatron"]} == {
                        "actor": 1e-6,
                        "critic": 1e-5,
                    }
        if "--rollout-data-dir" in args:
            assert Path(option("--rollout-data-dir")).is_dir()
            if platform == "npu":
                assert Path(option("--rollout-data-dir")).is_relative_to(root / "tmp")

        if "non_colocate_pp" in filename:
            assert option("--pipeline-model-parallel-size") == "2"
            assert option("--actor-num-gpus-per-node") == option("--rollout-num-gpus") == "2"
            assert "--colocate" not in args
            assert ("--load" in args) == (platform == "cuda")
            assert option("--num-rollout") == "2"
        elif "top_p_cp2" in filename:
            assert option("--context-parallel-size") == "2"
            assert option("--rollout-top-p") == "0.95"
            assert option("--rollout-top-k") == "20"
            assert option("--loss-mask-type") == "qwen3_5"
            assert "--colocate" in args
        elif "debug_train_dump" in filename:
            for flag in ("--tensor-model-parallel-size", "--pipeline-model-parallel-size", "--context-parallel-size"):
                assert option(flag) == "2"
            write_dump(option("--dump-details"))
        elif "ppo" in filename:
            assert option("--advantage-estimator") == "ppo"
            assert option("--tensor-model-parallel-size") == option("--context-parallel-size") == "2"
            assert option("--num-critic-only-steps") == ("2" if "critic_only" in filename else "1")
            assert option("--num-rollout") == "2"
            assert option("--rollout-max-response-len") == "8192"
            assert ("--optimizer-cpu-offload" in args) == (platform == "npu")
            separate = "disaggregate" in filename
            assert option("--actor-num-gpus-per-node") == option("--rollout-num-gpus") == ("4" if separate else "8")
            assert ("--colocate" in args) != separate
            if separate:
                assert option("--rollout-data-transport") == ("object-store" if platform == "npu" else "nixl")
            elif "critic_only" not in filename:
                assert option("--rollout-data-transport") == "straw"
        elif "parallel_check" in filename:
            assert option("--rollout-max-response-len") == "8192"
            assert ("--vllm-enable-deterministic-inference" in args) == (platform == "cuda")
            if "--save-debug-rollout-data" in args:
                Path(option("--save-debug-rollout-data").format(rollout_id=0)).touch()
            else:
                assert Path(option("--load-debug-rollout-data").format(rollout_id=0)).is_file()
            if "--ci-save-grad-norm" in args:
                Path(option("--ci-save-grad-norm")).touch()
            else:
                assert Path(option("--ci-load-grad-norm")).is_file()
            assert kwargs["num_gpus_per_node"] == int(option("--actor-num-gpus-per-node"))

    monkeypatch.setattr(U, "execute_train", record_train)
    module.prepare()
    downloads = [shlex.split(c) for c in commands if c.startswith("hf download")]
    assert len(downloads) == 2
    assert all(c[c.index("--local-dir") + 1].startswith(str(root) + "/") for c in downloads)
    assert not any("aime-2024" in c for c in commands)
    if platform == "npu":
        assert not any("torchrun" in c for c in commands)
    module.execute()
    assert all(not config.exists() for config in configs)
    if "parallel_check" not in filename:
        assert len(calls) == 1
        return
    assert len(calls) == 12
    saved = calls[0][calls[0].index("--save-debug-rollout-data") + 1]
    assert all(args[args.index("--load-debug-rollout-data") + 1] == saved for args in calls[1:])
    assert not Path(saved).parent.exists()
    for loss_index in range(2):
        runs = calls[loss_index * 6 : (loss_index + 1) * 6]
        assert all(("--calculate-per-token-loss" in args) == bool(loss_index) for args in runs)
        layouts = {
            tuple(
                int(args[args.index(flag) + 1])
                for flag in (
                    "--actor-num-gpus-per-node",
                    "--tensor-model-parallel-size",
                    "--pipeline-model-parallel-size",
                    "--context-parallel-size",
                )
            )
            for args in runs[1:]
        }
        assert layouts == {(2, 1, 1, 1), (8, 4, 1, 1), (8, 1, 4, 1), (8, 1, 1, 4), (8, 2, 2, 2)}


@pytest.mark.parametrize("broken", ["values", "position", "no_comparison"])
def test_debug_dump_still_rejects_invalid_alignment(tmp_path, broken):
    module = load_case("test_qwen2.5_0.5B_debug_train_dump_e2e.py")
    write_dump(tmp_path, broken=broken)
    with pytest.raises(AssertionError):
        module._verify(str(tmp_path))


def test_nixl_remains_an_explicit_probe_on_npu(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS["npu"])
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path))
    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(shlex.split(kwargs["train_args"])))
    module = load_case("test_qwen3_4B_ppo_disaggregate.py")
    module.execute(transport="nixl")
    assert calls[0][calls[0].index("--rollout-data-transport") + 1] == "nixl"
