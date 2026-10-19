"""Record shared E2E launches with CPU fixtures; never start Ray or training."""

import importlib.util
import json
import math
import shlex
from pathlib import Path

import pytest
import torch

from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils.launch import PLATFORMS
from vime.utils.types import Sample

NUM_GPUS = 0
TESTS = Path(__file__).resolve().parents[1]


def load_case(filename):
    spec = importlib.util.spec_from_file_location("shared_hardware_case", TESTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_asset_root_defaults_and_explicit_override(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[platform])
    monkeypatch.delenv("VIME_TEST_ROOT", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert U.get_test_root() == (tmp_path / "hf" if platform == "npu" else Path("/root"))
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path / "override"))
    assert U.get_test_root() == tmp_path / "override"
    temporary = U.get_test_tmp_dir()
    if platform == "npu":
        assert Path(temporary).is_dir()
        assert Path(temporary).parent == tmp_path / "override"
    else:
        assert temporary is None


CASES = [
    ("test_qwen2.5_0.5B_fully_async_short.py", {}),
    ("test_qwen2.5_0.5B_pipeline_rl.py", {"transport": "nccl", "flush_cache_interval": 0}),
    ("test_qwen2.5_0.5B_pipeline_rl.py", {"transport": "disk", "flush_cache_interval": 0}),
    ("test_qwen2.5_0.5B_pipeline_rl.py", {"transport": "nccl", "flush_cache_interval": 2}),
    ("test_full_disk_weight_update.py", {}),
    ("test_qwen2.5_0.5B_fanout_short.py", {}),
    ("test_qwen2.5_0.5B_score_centering.py", {"top_p": 1.0}),
    ("test_qwen2.5_0.5B_score_centering.py", {"top_p": 0.95}),
]


def write_full_checkpoints(directory, num_rollout):
    for version in range(1, num_rollout + 2):
        path = Path(directory) / f"weight_v{version:06d}"
        path.mkdir(parents=True)
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
        (path / "model.safetensors").write_bytes(b"fixture")


@pytest.mark.parametrize("platform", ["cuda", "npu"])
@pytest.mark.parametrize("filename,options", CASES)
def test_shared_cases_keep_their_assertions_and_use_platform_paths(monkeypatch, tmp_path, platform, filename, options):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[platform])
    monkeypatch.setattr(torch.version, "hip", None)
    root = tmp_path / "assets with spaces"
    monkeypatch.setenv("VIME_TEST_ROOT", str(root))
    monkeypatch.setenv("VIME_FANOUT_TEST_COUNTER_FILE", str(tmp_path / "fanout.log"))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("VIME_TEST_NUM_ROLLOUT", "2" if platform == "npu" else "1")
    commands = []
    monkeypatch.setattr(U, "exec_command", commands.append)
    module = load_case(filename)
    calls = []

    def record_train(**kwargs):
        calls.append(kwargs)
        args = shlex.split(kwargs["train_args"])

        def option(flag):
            return args[args.index(flag) + 1]

        assert option("--hf-checkpoint") == str(root / "models" / module.MODEL_NAME)
        ref = option("--ref-load")
        if filename == "test_full_disk_weight_update.py" and platform == "cuda":
            assert ref == module.TORCH_DIST_CKPT
        else:
            assert ref == option("--hf-checkpoint")
        assert option("--prompt-data").startswith(str(root / "datasets") + "/")
        assert ("--use-flash-attn" in args) == (platform == "npu")
        assert ("--no-gradient-accumulation-fusion" in args) == (platform == "npu")
        if platform == "npu":
            assert json.loads(option("--vllm-additional-config")) == {"weight_nz_mode": 0}
        assert kwargs["num_gpus_per_node"] == module.NUM_GPUS
        if "--rollout-data-dir" in args and platform == "npu":
            assert Path(option("--rollout-data-dir")).is_relative_to(root / "tmp")
        if "pipeline_rl" in filename:
            assert option("--num-rollout") == "3"
            assert option("--rollout-num-gpus") == "3"
            assert option("--vllm-max-model-len") == "32768"
            assert ("--update-weight-transport" in args) == (options["transport"] == "disk")
            assert option("--flush-cache-interval") == str(options["flush_cache_interval"])
            before = {"version": "1", "tokens": 16}
            after = {"version": "2", "tokens": 40}
            next_after = {
                "version": "3",
                "tokens": 64,
                "finish_reason": "abort" if options["flush_cache_interval"] == 2 else None,
            }
            probes = {
                "weights_changed": True,
                "continuous": [{"before": before, "after": after}] * 3,
                "next_update": [{"before": after, "after": next_after, "serving_version": "3"}] * 3,
            }
            Path(kwargs["extra_env_vars"]["VIME_PIPELINE_RL_PROBE_FILE"]).write_text(json.dumps(probes))
            checkpoint = Path(option("--save")) / "rollout"
            checkpoint.mkdir(parents=True)
            (checkpoint / "committed_2.json").write_text("{}")
        elif "full_disk" in filename:
            assert option("--max-tokens-per-gpu") == ("2048" if platform == "npu" else "9216")
            assert option("--rollout-max-response-len") == "1024"
            assert option("--entropy-coef") == "0.01"
            if platform == "npu":
                assert option("--log-probs-chunk-size") == "128"
                assert option("--num-rollout") == "2"
            else:
                assert "--log-probs-chunk-size" not in args
            write_full_checkpoints(option("--update-weight-disk-dir"), int(option("--num-rollout")))
        elif "fanout" in filename:
            assert option("--n-samples-per-prompt") == "1"
            Path(kwargs["extra_env_vars"]["VIME_FANOUT_TEST_COUNTER_FILE"]).write_text("sample\n" * 8)
        elif "score_centering" in filename:
            assert option("--rollout-top-p") == str(options["top_p"])
            sample = Sample(
                tokens=[0, 1],
                response_length=1,
                loss_mask=[1],
                rollout_log_probs=[-math.log(128)],
                rollout_topk_token_ids=[list(range(128))],
                rollout_topk_log_probs=[[-math.log(128)] * 128],
                rollout_top_p_token_ids=[1],
                rollout_top_p_token_offsets=[0, 1],
                rollout_top_p_log_probs=[-math.log(128)],
            )
            for rollout_id in range(2):
                torch.save(
                    {"samples": [sample.to_dict()]}, option("--save-debug-rollout-data").format(rollout_id=rollout_id)
                )
                torch.save(0.1, option("--ci-save-grad-norm").format(rollout_id=rollout_id, step_id=0))
        else:
            assert option("--num-rollout") == "2"
            assert option("--rollout-max-response-len") == "8192"

    monkeypatch.setattr(U, "execute_train", record_train)
    module.prepare()
    downloads = [shlex.split(command) for command in commands if command.startswith("hf download")]
    assert len(downloads) == 2
    for command in downloads:
        assert command[command.index("--local-dir") + 1].startswith(str(root) + "/")
    if platform == "npu":
        assert not any("torchrun" in command for command in commands)
    module.execute(**options)
    assert len(calls) == 1


@pytest.mark.parametrize("broken", ["missing_update", "missing_shard", "delta"])
def test_full_disk_rejects_incomplete_publications(tmp_path, broken):
    module = load_case("test_full_disk_weight_update.py")
    write_full_checkpoints(tmp_path, 2)
    last = tmp_path / "weight_v000003"
    if broken == "missing_update":
        last.rename(tmp_path / "incomplete")
    elif broken == "missing_shard":
        (last / "model.safetensors").unlink()
    else:
        index = last / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        data["metadata"] = {"delta_encoding": "xor"}
        index.write_text(json.dumps(data))
    with pytest.raises((AssertionError, FileNotFoundError)):
        module.verify_checkpoints(tmp_path, 2)
