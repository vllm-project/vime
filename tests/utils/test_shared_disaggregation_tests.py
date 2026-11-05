"""Final migration batch contracts without hardware or external services."""

import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from vime.platforms import get_platform
from vime.utils.external_utils import command_utils as U
from vime.utils.external_utils.launch import PLATFORMS

NUM_GPUS = 0
TESTS = Path(__file__).resolve().parents[1]


def load(monkeypatch, filename):
    name = filename.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, TESTS / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def option(args, flag):
    return args[max(i for i, value in enumerate(args) if value == flag) + 1]


@pytest.fixture(params=["cuda", "npu"])
def platform(request, monkeypatch, tmp_path):
    monkeypatch.setattr(U, "current_platform", lambda: PLATFORMS[request.param])
    monkeypatch.setenv("VIME_TEST_ROOT", str(tmp_path / "assets with spaces"))
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "0")
    monkeypatch.setenv("VIME_TEST_ENABLE_EVAL", "0")
    monkeypatch.setenv("VIME_TEST_USE_DEEPEP", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    return get_platform(request.param)


@pytest.mark.parametrize(
    "filename", ["test_qwen2.5_vl_3B_ep_disaggregation.py", "test_qwen3.6_35B_A3B_pd_mooncake.py"]
)
def test_pd_public_model_download_retains_model_revision_and_cache(platform, monkeypatch, filename):
    module = load(monkeypatch, filename)
    calls = []
    monkeypatch.setattr(U, "hf_download_public_model", lambda repo, **kwargs: calls.append((repo, kwargs)))
    monkeypatch.setattr(U, "hf_download_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(U, "convert_checkpoint", lambda **kwargs: None)
    module.prepare()
    assert calls[0][0] == f"Qwen/{module.MODEL_NAME}"
    assert calls[0][1]["dir_dst"] == (module.MODEL_PATH if "vl_3B" in filename else module.MODEL_DIR)
    assert calls[0][1].get("revision") == getattr(module, "MODEL_REVISION", None)


@pytest.mark.parametrize("filename", ["test_glm4.7_30B_A3B_pd_mooncake.py", "test_qwen3.6_35B_A3B_pd_mooncake.py"])
def test_large_pd_cases_keep_topologies_and_mtp_with_owned_scratch(platform, monkeypatch, filename):
    module = load(monkeypatch, filename)
    scratch = []

    def train(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert kwargs["num_gpus_per_node"] == 8
        assert option(args, "--tensor-model-parallel-size") == "2"
        assert option(args, "--context-parallel-size") == "2"
        assert option(args, "--num-rollout") == "2"
        assert json.loads(option(args, "--vllm-speculative-config"))["num_speculative_tokens"] == 4
        assert "--colocate" in args and "--ci-test" in args
        assert option(args, "--hf-checkpoint") == str(module.MODEL_DIR)
        assert option(args, "--ref-load") == str(
            module.MODEL_DIR if platform.is_npu else module.MODEL_DIR.parent / (module.MODEL_NAME + "_torch_dist")
        )
        if "glm" in filename:
            path = Path(option(args, "--vllm-config"))
            config = yaml.safe_load(path.read_text())
            assert [(group["worker_type"], group["num_gpus"]) for group in config["vllm"][0]["server_groups"]] == [
                ("prefill", 4),
                ("decode", 4),
            ]
            assert option(args, "--pipeline-model-parallel-size") == "2"
            scratch.append(path.parent)
        else:
            assert option(args, "--prefill-num-servers") == "1"
            assert option(args, "--rollout-max-response-len") == "16384"
            assert option(args, "--expert-model-parallel-size") == "8"
            assert ("--moe-enable-deepep" in args) is not platform.is_npu
            scratch.append(Path(option(args, "--rollout-data-dir")).parent)
            assert "--eval-prompt-data" not in args
        if platform.is_npu:
            assert option(args, "--log-probs-chunk-size") == "128"
            assert option(args, "--max-tokens-per-gpu") == "2048"
        assert scratch[-1].exists()

    monkeypatch.setattr(U, "execute_train", train)
    module.execute()
    assert scratch and not any(path.exists() for path in scratch)


@pytest.mark.parametrize("failure", [False, True])
def test_external_pd_owns_processes_and_restores_device_visibility(platform, monkeypatch, failure, tmp_path):
    module = load(monkeypatch, "test_qwen3_4B_external_pd.py")
    monkeypatch.setattr(module, "current_platform", lambda: platform)
    monkeypatch.setattr(module, "_get_external_host", lambda: "127.0.0.1")
    visible = "7,6,5,4,3,2,1,0"
    monkeypatch.setenv(platform.ray.visible_devices_env, visible)
    monkeypatch.setenv("VIME_TEST_UPDATE_MODE", "delta")
    # Keep the test's local-only staging policy while using a disposable fixture.
    temporary = U.test_work_directory
    monkeypatch.setattr(
        U,
        "test_work_directory",
        lambda prefix, **kw: (
            temporary(prefix, **{**kw, "parent": str(tmp_path)})
            if kw.get("parent") == "/tmp"
            else temporary(prefix, **kw)
        ),
    )
    started, stopped, scratch = [], [], []

    def launch(**kwargs):
        assert kwargs["gpus"] == (["3"] if kwargs["disaggregation_mode"] == "prefill" else ["2"])
        assert os.environ[platform.ray.visible_devices_env] == "7,6,5,4"
        path = Path(kwargs["log_path"])
        path.write_text("fixture log")
        scratch.append(path.parent)
        process = Mock(pid=len(started) + 100)
        started.append(process)
        return process

    def stop(pid):
        assert scratch[0].exists()
        stopped.append(pid)

    def train(**kwargs):
        args = shlex.split(kwargs["train_args"])
        assert kwargs["num_gpus_per_node"] == 4
        assert "--rollout-external-engine-addrs" in args
        assert option(args, "--update-weight-mode") == "delta"
        assert option(args, "--entropy-coef") == "0.01"
        kwargs["before_ray_job_submit"]()
        if failure:
            raise RuntimeError("fixture training failure")
        path = Path(option(args, "--update-weight-disk-dir")) / "weight_v1"
        path.mkdir(parents=True)
        (path / "weights.safetensors").write_bytes(b"fixture")

    monkeypatch.setattr(module, "_launch_vllm_server", launch)
    monkeypatch.setattr(module, "kill_process_tree", stop)
    monkeypatch.setattr(U, "execute_train", train)
    if failure:
        with pytest.raises(RuntimeError, match="fixture training failure"):
            module.execute()
    else:
        module.execute()
    assert stopped == [100, 101]
    assert os.environ[platform.ray.visible_devices_env] == visible
    assert not any(path.exists() for path in scratch)


@pytest.mark.parametrize("side", ["prefill", "decode"])
def test_external_server_uses_platform_kv_and_weight_backends(platform, monkeypatch, side, tmp_path):
    module = load(monkeypatch, "test_qwen3_4B_external_pd.py")
    monkeypatch.setattr(module, "current_platform", lambda: platform)
    monkeypatch.setattr(module, "is_port_available", lambda port: True)
    launches = []
    process = Mock(pid=123)
    process.poll.return_value = None
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda command, **kwargs: launches.append((command, kwargs)) or process
    )
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: SimpleNamespace(status=200))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    module._launch_vllm_server(
        gpus=["5"],
        port=13150,
        tp=1,
        log_path=str(tmp_path / "server.log"),
        disaggregation_mode=side,
        disaggregation_bootstrap_port=13200,
    )
    command, kwargs = launches[0]
    config = json.loads(option(command, "--kv-transfer-config"))
    assert config["kv_connector"] == ("MooncakePullConnector" if platform.is_npu else "NixlConnector")
    assert config["kv_role"] == ("kv_producer" if side == "prefill" else "kv_consumer")
    assert kwargs["env"][platform.ray.visible_devices_env] == "5"
    assert kwargs["env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] == ("13201" if platform.is_npu else "13200")
    assert json.loads(option(command, "--weight-transfer-config"))["backend"] == (
        "hccl" if platform.is_npu else "nccl"
    )
    if platform.is_npu:
        assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]


def test_epd_config_preserves_encoder_features_and_kv_roles(platform, monkeypatch, tmp_path):
    module = load(monkeypatch, "test_qwen2.5_vl_3B_ep_disaggregation.py")
    monkeypatch.setattr(module, "current_platform", lambda: platform)
    monkeypatch.setattr(module, "vllm_parse_args", lambda: SimpleNamespace())
    path = module._write_config(("encoder", "prefill", "decode"), tmp_path)
    groups = yaml.safe_load(Path(path).read_text())["vllm"][0]["server_groups"]
    assert [group["worker_type"] for group in groups] == ["encoder", "prefill", "decode"]
    args = module._make_args(path, ("encoder", "prefill", "decode"))
    assert args.rollout_num_gpus == 3
    assert args.vllm_enable_deterministic_inference is not platform.is_npu
    assert args.debug_rollout_only and not args.colocate


def test_agent_requires_sandbox_before_model_download_on_npu(platform, monkeypatch):
    module = load(monkeypatch, "test_agent_sunabako_codex_e2e.py")
    monkeypatch.delenv("SUNABAKO_CLUSTER", raising=False)
    monkeypatch.delenv("SUNABAKO_IMAGES", raising=False)
    if platform.is_npu:
        with pytest.raises(RuntimeError, match="x86_64 sandbox cluster"):
            module.prepare_assets()
    else:
        module.check_runtime_prerequisites()
    monkeypatch.setenv("SUNABAKO_CLUSTER", "/fixture/cluster.json")
    monkeypatch.setenv("SUNABAKO_IMAGES", "/fixture/images.json")
    module.check_runtime_prerequisites()
