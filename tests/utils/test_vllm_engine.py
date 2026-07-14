"""CPU unit tests for ``vime.backends.vllm_utils.vllm_engine``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import requests
import torch

_unit_stubs.install_vllm_cli_stubs()

from vime.backends.vllm_utils import vllm_engine as mod

NUM_GPUS = 0


@pytest.fixture(autouse=True)
def _patch_vllm_server_fields(monkeypatch):
    """Stub _VLLM_SERVER_FIELDS so tests don't need a real vllm install."""
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset())


@pytest.fixture
def vllm_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_external=True,
        hf_checkpoint="/tmp/model",
        vllm_router_ip=None,
        vllm_router_port=None,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=4,
        colocate=False,
        debug_rollout_only=False,
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        use_critic=False,
        critic_num_gpus_per_node=0,
        critic_num_nodes=0,
        seed=1234,
        fp16=False,
        offload_rollout=False,
        use_rollout_routing_replay=False,
        vllm_pipeline_parallel_size=1,
        vllm_data_parallel_size=1,
        vllm_dp_size=1,
    )


@pytest.fixture
def vllm_engine(vllm_args):
    from vime.backends.vllm_utils.vllm_engine import VLLMEngine

    engine = VLLMEngine(vllm_args, rank=0)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 8765
    return engine


class _MockResponse:
    def __init__(self, *, json_data: dict | None = None, text: str = "", status_code: int = 200):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code
        # Model requests.Response.content (raw body bytes) so _response_json's empty-body
        # handling (empty 200 -> {"ok": True}) is actually exercised. A JSON body is non-empty;
        # text-only/empty bodies use the given text (b"" when empty).
        self.content = json.dumps(json_data).encode() if json_data is not None else text.encode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict:
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


@pytest.mark.unit
def test_normalize_vllm_wake_tags_drops_unsupported():
    assert mod._normalize_vllm_wake_tags(["weights", "cuda_graph", "kv_cache"]) == ["weights", "kv_cache"]


@pytest.mark.unit
def test_normalize_vllm_wake_tags_empty_becomes_none():
    assert mod._normalize_vllm_wake_tags(["cuda_graph"]) is None


@pytest.mark.unit
def test_launch_config_single_node(vllm_args):
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 4
    vllm_args.vllm_pipeline_parallel_size = 1
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert sa["nnodes"] == 1
    assert sa["node_rank"] == 0
    assert sa["_tp_size"] == 4


@pytest.mark.unit
def test_compute_server_args_preserves_non_topology_vllm_flags(vllm_args, monkeypatch):
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset({"server_concurrency", "tool_call_parser"}))
    vllm_args.vllm_server_concurrency = 256
    vllm_args.vllm_tool_call_parser = "qwen3_coder"
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert sa["server_concurrency"] == 256
    assert sa["tool_call_parser"] == "qwen3_coder"


@pytest.mark.unit
def test_compute_server_args_ignores_unrecognized_vllm_attrs(vllm_args, monkeypatch):
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset({"server_concurrency"}))
    vllm_args.vllm_nonexistent_flag = "keep-out"
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert "nonexistent_flag" not in sa


@pytest.mark.unit
def test_launch_config_multi_node_ranks(vllm_args):
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 16
    vllm_args.vllm_pipeline_parallel_size = 2
    sa0, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr="10.0.0.1:15000", host="10.0.0.1", port=8000)
    sa1, _ = mod._compute_server_args(vllm_args, rank=1, dist_init_addr="10.0.0.1:15000", host="10.0.0.2", port=8000)
    assert sa0["nnodes"] == 2
    assert sa0["node_rank"] == 0
    assert sa1["node_rank"] == 1


@pytest.mark.unit
def test_distributed_flags_only_when_multi_node(vllm_args):
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 4
    vllm_args.vllm_pipeline_parallel_size = 1
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert "master_addr" not in sa

    vllm_args.rollout_num_gpus_per_engine = 16
    vllm_args.vllm_pipeline_parallel_size = 2
    sa_multi, _ = mod._compute_server_args(
        vllm_args, rank=1, dist_init_addr="10.0.0.2:16000", host="10.0.0.2", port=8000
    )
    assert sa_multi["nnodes"] == 2
    assert sa_multi["node_rank"] == 1
    assert sa_multi["master_addr"] == "10.0.0.2"
    assert sa_multi.get("headless") is True
    assert sa_multi.get("data_parallel_backend") == "mp"
    assert sa_multi.get("distributed_executor_backend") == "mp"


@pytest.mark.unit
def test_compute_server_args_applies_worker_type_and_bootstrap_port(vllm_args):
    sa_prefill, _ = mod._compute_server_args(
        vllm_args,
        rank=0,
        dist_init_addr=None,
        host="127.0.0.1",
        port=8000,
        worker_type="prefill",
        disaggregation_bootstrap_port=12345,
    )
    assert sa_prefill["kv_transfer_config"] == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
    }

    sa_decode, _ = mod._compute_server_args(
        vllm_args,
        rank=0,
        dist_init_addr=None,
        host="127.0.0.1",
        port=8000,
        worker_type="decode",
    )
    assert sa_decode["kv_transfer_config"] == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
    }


@pytest.mark.unit
def test_compute_server_args_prefill_requires_bootstrap_port(vllm_args):
    with pytest.raises(AssertionError, match="disaggregation_bootstrap_port"):
        mod._compute_server_args(
            vllm_args,
            rank=0,
            dist_init_addr=None,
            host="127.0.0.1",
            port=8000,
            worker_type="prefill",
        )


@pytest.mark.unit
def test_compute_server_args_applies_rollout_and_dtype_flags(vllm_args):
    vllm_args.use_rollout_routing_replay = True
    vllm_args.fp16 = True
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert sa["enable_return_routed_experts"] is True
    assert sa["dtype"] == "float16"


@pytest.mark.unit
def test_compute_server_args_applies_max_model_len_from_rollout_context(vllm_args):
    vllm_args.rollout_max_context_len = 8192
    vllm_args.vllm_max_model_len = None
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert sa["max_model_len"] == 8192


@pytest.mark.unit
def test_compute_server_args_model_path_override_wins(vllm_args, monkeypatch):
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset({"model", "server_concurrency"}))
    sa, _ = mod._compute_server_args(
        vllm_args,
        rank=0,
        dist_init_addr=None,
        host="127.0.0.1",
        port=8000,
        vllm_overrides={"model_path": "/tmp/override", "server-concurrency": 123},
    )
    assert sa["model"] == "/tmp/override"
    assert sa["server_concurrency"] == 123


@pytest.mark.unit
def test_compute_server_args_external_check_fields_skip_orchestration_fields(vllm_args):
    sa, check_fields = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert "model" not in check_fields
    assert "host" not in check_fields
    assert "port" not in check_fields
    assert "nnodes" in check_fields
    assert "node_rank" in check_fields
    assert "weight_transfer_config" in check_fields
    assert sa["weight_transfer_config"] == {"backend": "nccl"}


@pytest.mark.unit
def test_build_vllm_subprocess_env_colocate(vllm_args, monkeypatch):
    vllm_args.colocate = True
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = mod._build_subprocess_env(
        {
            "_args": vllm_args,
            "_visible_devices": "0,1",
        }
    )
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" in env
    assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert "PYTHONPATH" in env


@pytest.mark.unit
def test_build_vllm_subprocess_env_sets_batch_invariant_when_deterministic(vllm_args, monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    vllm_args.vllm_enable_deterministic_inference = True
    env = mod._build_subprocess_env({"_args": vllm_args, "_visible_devices": "0"})
    assert env["VLLM_BATCH_INVARIANT"] == "1"


@pytest.mark.unit
def test_build_vllm_subprocess_env_no_batch_invariant_by_default(vllm_args, monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    vllm_args.vllm_enable_deterministic_inference = False
    env = mod._build_subprocess_env({"_args": vllm_args, "_visible_devices": "0"})
    assert "VLLM_BATCH_INVARIANT" not in env


@pytest.mark.unit
def test_build_vllm_subprocess_env_sets_disaggregation_side_channel(vllm_args):
    env = mod._build_subprocess_env(
        {
            "_args": vllm_args,
            "_visible_devices": "0",
            "_worker_type": "prefill",
            "_disaggregation_bootstrap_port": 29999,
            "node_rank": 0,
            "host": "10.0.0.8",
        }
    )
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.8"
    assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "29999"


@pytest.mark.unit
def test_compute_server_args_adds_sleep_mode_for_offload_rollout(vllm_args):
    vllm_args.offload_rollout = True
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert sa.get("enable_sleep_mode") is True
    assert vllm_args.vllm_enable_sleep_mode is True


@pytest.mark.unit
def test_compute_server_args_no_sleep_mode_from_colocate(vllm_args):
    vllm_args.colocate = True
    vllm_args.offload_rollout = False
    sa, _ = mod._compute_server_args(vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000)
    assert "enable_sleep_mode" not in sa
    assert not getattr(vllm_args, "vllm_enable_sleep_mode", False)


@pytest.mark.unit
def test_get_base_gpu_id_colocate(vllm_args):
    vllm_args.colocate = True
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 4
    assert mod.get_base_gpu_id(vllm_args, rank=1) == 4


@pytest.mark.unit
def test_start_weight_update_posts_four_phase_endpoint(vllm_engine, monkeypatch):
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict):
        calls.append((endpoint, payload))
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine.start_weight_update(is_checkpoint_format=True)

    assert result == {"ok": True}
    assert len(calls) == 1
    assert calls[0][0] == "start_weight_update"
    assert calls[0][1] == {"is_checkpoint_format": True}


@pytest.mark.unit
def test_start_draft_weight_update_posts_empty_body(vllm_engine, monkeypatch):
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict):
        calls.append((endpoint, payload))
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine.start_draft_weight_update()

    assert result == {"ok": True}
    assert calls == [("start_draft_weight_update", {})]


@pytest.mark.unit
def test_finish_weight_update_posts_empty_body(vllm_engine, monkeypatch):
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict):
        calls.append((endpoint, payload))
        return {"done": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine.finish_weight_update()

    assert result == {"done": True}
    assert calls == [("finish_weight_update", {})]


@pytest.mark.unit
def test_update_weights_from_tensor_posts_ipc_payload_and_records_version(vllm_engine, monkeypatch):
    posted: list[tuple[str, dict]] = []

    def fake_post(endpoint: str, payload: dict):
        posted.append((endpoint, payload))
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)
    assert vllm_engine._weight_version is None

    vllm_engine.update_weights_from_tensor(
        names=["layer.0.weight"],
        dtype_names=["float32"],
        shapes=[[2, 2]],
        ipc_handles=[{"uuid-gpu0": ("rebuild_fn", (1, 2, 3))}],
        weight_version="42",
    )

    assert posted[0][0] == "update_weights"
    sent = posted[0][1]["update_info"]
    # ipc_handles got cloudpickle'd into ipc_handles_pickled
    assert "ipc_handles" not in sent
    assert isinstance(sent["ipc_handles_pickled"], str)
    assert sent["names"] == ["layer.0.weight"]
    assert sent["shapes"] == [[2, 2]]
    # version recorded after POST success
    assert vllm_engine._weight_version == "42"


@pytest.mark.unit
def test_update_weights_from_tensor_does_not_advance_version_on_failure(vllm_engine, monkeypatch):
    """POST failure must not advance _weight_version (else a retry would skip the resync)."""

    def fake_post_fail(endpoint: str, payload: dict) -> dict:
        raise RuntimeError("simulated POST failure")

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post_fail)

    vllm_engine._weight_version = "old"
    with pytest.raises(RuntimeError, match="simulated POST failure"):
        vllm_engine.update_weights_from_tensor(
            names=[], dtype_names=[], shapes=[], ipc_handles=[], weight_version="new"
        )
    assert vllm_engine._weight_version == "old"


@pytest.mark.unit
def test_get_weight_version_returns_recorded_version(vllm_engine):
    vllm_engine._weight_version = "7"
    assert vllm_engine.get_weight_version() == "7"


@pytest.mark.unit
def test_get_weight_version_raises_when_unset(vllm_engine):
    """Unrecorded version is a hard error — no silent /v1/models fallback."""
    assert vllm_engine._weight_version is None
    with pytest.raises(RuntimeError, match="before any successful weight transfer"):
        vllm_engine.get_weight_version()


@pytest.mark.unit
def test_get_weight_version_worker_rank_returns_none_without_raise(vllm_engine):
    """Worker ranks short-circuit (matches the class-wide idiom)."""
    vllm_engine.node_rank = 1
    vllm_engine._weight_version = None
    assert vllm_engine.get_weight_version() is None


@pytest.mark.unit
def test_update_weights_from_distributed_posts_update_weights_without_checkpoint_flag(vllm_engine, monkeypatch):
    calls: list[dict] = []

    def fake_make_request(endpoint: str, payload: dict) -> dict:
        calls.append(payload.get("update_info", payload))
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_make_request)

    names = ["layer.0.weight"]
    dtypes = [torch.float32]
    shapes = [torch.Size([2, 2])]

    vllm_engine.update_weights_from_distributed(
        names,
        dtypes,
        shapes,
        group_name="vime-pp_0",
        weight_version="7",
        packed=True,
    )

    assert len(calls) == 1
    info = calls[0]
    assert info["names"] == names
    assert info["dtype_names"] == ["float32"]
    assert info["shapes"] == [[2, 2]]
    assert info["packed"] is True
    assert "is_checkpoint_format" not in info
    assert vllm_engine._weight_version == "7"


@pytest.mark.unit
def test_get_url_ipv6_host(vllm_engine):
    vllm_engine.server_host = "[2001:db8::1]"
    assert vllm_engine.get_url() == "http://[2001:db8::1]:8765"


@pytest.mark.unit
def test_get_base_gpu_id_with_critic_offset(vllm_args):
    vllm_args.colocate = False
    vllm_args.debug_rollout_only = False
    vllm_args.actor_num_gpus_per_node = 4
    vllm_args.actor_num_nodes = 1
    vllm_args.use_critic = True
    vllm_args.critic_num_gpus_per_node = 2
    vllm_args.critic_num_nodes = 1
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 2
    # actor 4 + critic 2 + rank0*2 = 6
    assert mod.get_base_gpu_id(vllm_args, rank=0) == 6


@pytest.mark.unit
def test_resume_memory_occupation_wake_tags_query(vllm_engine, monkeypatch):
    seen: list[tuple] = []

    def fake_post(url, *, params=None, timeout=30, json=None):
        seen.append((url, params, timeout, json))
        return _MockResponse(json_data={"ok": True})

    vllm_args = vllm_engine.args
    vllm_args.vllm_enable_sleep_mode = True
    monkeypatch.setattr(mod.requests, "post", fake_post)

    vllm_engine.resume_memory_occupation(tags=["weights", "cuda_graph"])

    assert len(seen) == 1
    assert seen[0][1] == [("tags", "weights")]


@pytest.mark.unit
def test_resume_memory_occupation_returns_none_for_unsupported_tags(vllm_engine, monkeypatch):
    seen: list[tuple] = []

    def fake_post(url, *, params=None, timeout=30, json=None):
        seen.append((url, params, timeout, json))
        return _MockResponse(json_data={"ok": True})

    monkeypatch.setattr(mod.requests, "post", fake_post)

    assert vllm_engine.resume_memory_occupation(tags=["cuda_graph"]) == {"ok": True}
    assert seen[0][1] is None


@pytest.mark.unit
def test_release_memory_occupation_flushes_then_posts_sleep(vllm_engine, monkeypatch):
    calls: list[str] = []

    def fake_flush_cache():
        calls.append("flush_cache")

    def fake_post(url, *, params=None, timeout=30, json=None):
        calls.append(url)
        assert params == {"level": 2}
        assert timeout == 30
        assert json is None
        return _MockResponse(json_data={"ok": True, "sleep_mode": True})

    vllm_engine.args.vllm_enable_sleep_mode = False
    monkeypatch.setattr(vllm_engine, "flush_cache", fake_flush_cache)
    monkeypatch.setattr(mod.requests, "post", fake_post)

    assert vllm_engine.release_memory_occupation(level=2) == {"ok": True, "sleep_mode": True}
    assert calls == ["flush_cache", "http://127.0.0.1:8765/sleep"]


@pytest.mark.unit
def test_resume_memory_occupation_posts_wake_even_when_sleep_disabled(vllm_engine, monkeypatch):
    seen: list[tuple] = []

    def fake_post(url, *, params=None, timeout=30, json=None):
        seen.append((url, params, timeout, json))
        return _MockResponse(json_data={"ok": True, "sleep_mode": True})

    vllm_engine.args.vllm_enable_sleep_mode = False
    monkeypatch.setattr(mod.requests, "post", fake_post)

    assert vllm_engine.resume_memory_occupation() == {"ok": True, "sleep_mode": True}
    assert seen == [("http://127.0.0.1:8765/wake_up", None, 30, None)]


@pytest.mark.unit
def test_init_weights_update_group_posts_init_info(vllm_engine, monkeypatch):
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict):
        calls.append((endpoint, payload))
        return {"initialized": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine.init_weights_update_group(
        "127.0.0.1",
        29500,
        rank_offset=1,
        world_size=4,
        group_name="unused",
        backend="nccl",
    )

    assert result == {"initialized": True}
    assert len(calls) == 1
    assert calls[0][0] == "init_weight_transfer_engine"
    assert calls[0][1]["init_info"]["master_address"] == "127.0.0.1"
    assert calls[0][1]["init_info"]["rank_offset"] == 1


@pytest.mark.unit
def test_update_weights_from_disk_posts_collective_rpc(vllm_engine, monkeypatch):
    seen: list[tuple] = []

    def fake_post(url, *, params=None, timeout=30, json=None):
        seen.append((url, params, timeout, json))
        return _MockResponse(json_data={"reloaded": True})

    monkeypatch.setattr(mod.requests, "post", fake_post)

    assert vllm_engine.update_weights_from_disk("/tmp/model", weight_version="8") == {"reloaded": True}
    assert seen[0][0] == "http://127.0.0.1:8765/collective_rpc"
    assert seen[0][3]["method"] == "reload_weights"
    assert vllm_engine.get_weight_version() == "8"


@pytest.mark.unit
def test_profile_worker_rank_skips_http(vllm_engine, monkeypatch):
    vllm_engine.node_rank = 1
    monkeypatch.setattr(mod.requests, "post", lambda *args, **kwargs: pytest.fail("unexpected HTTP request"))

    assert vllm_engine.start_profile() is None
    assert vllm_engine.stop_profile() is None


@pytest.mark.unit
def test_update_weights_from_disk_surfaces_http_error(vllm_engine, monkeypatch):
    def fake_post(url, *, params=None, timeout=30, json=None):
        return _MockResponse(text="boom", status_code=500)

    monkeypatch.setattr(mod.requests, "post", fake_post)
    with pytest.raises(requests.exceptions.HTTPError):
        vllm_engine.update_weights_from_disk("/tmp/model")


@pytest.mark.unit
def test_resolve_parallel_sizes_is_per_engine_not_global(vllm_args):
    # The global flag is 1, but THIS engine has 2 GPUs → tp must be 2 (per-engine), not 1.
    # A stale global vllm_tp_size must NOT shadow the per-engine value.
    vllm_args.rollout_num_gpus_per_engine = 1
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_tp_size = 1  # stale global; must be ignored now
    tp, pp, dp = mod._resolve_parallel_sizes(vllm_args, gpus_per_engine=2)
    assert (tp, pp) == (2, 1)


@pytest.mark.unit
def test_launch_config_heterogeneous_per_group_tp(vllm_args):
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 1
    vllm_args.vllm_pipeline_parallel_size = 1
    sa, _ = mod._compute_server_args(
        vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000, num_gpus_per_engine=2
    )
    assert sa["_tp_size"] == 2
    assert sa["nnodes"] == 1


@pytest.mark.unit
def test_resolve_parallel_sizes_dp_consumes_gpus(vllm_args):
    # vLLM DP consumes GPUs (total = tp * pp * dp), so tp = gpus // (pp * dp).
    # dp=2, pp=1, 4 GPUs/engine → tp=2.
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_data_parallel_size = 2
    vllm_args.vllm_dp_size = 2
    tp, pp, dp = mod._resolve_parallel_sizes(vllm_args, gpus_per_engine=4)
    assert (tp, pp) == (2, 1)


@pytest.mark.unit
def test_resolve_parallel_sizes_dp_and_pp_combined(vllm_args):
    # dp=2, pp=2, 8 GPUs/engine → tp = 8 // (2*2) = 2.
    vllm_args.vllm_pipeline_parallel_size = 2
    vllm_args.vllm_data_parallel_size = 2
    vllm_args.vllm_dp_size = 2
    tp, pp, dp = mod._resolve_parallel_sizes(vllm_args, gpus_per_engine=8)
    assert (tp, pp) == (2, 2)


@pytest.mark.unit
def test_resolve_parallel_sizes_rejects_indivisible_dp(vllm_args):
    # gpus_per_engine not divisible by pp*dp must raise (fail fast, not desync the rendezvous).
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_data_parallel_size = 2
    vllm_args.vllm_dp_size = 2
    with pytest.raises(ValueError, match="divisible"):
        mod._resolve_parallel_sizes(vllm_args, gpus_per_engine=3)


@pytest.mark.unit
def test_make_request_short_circuits_on_headless(vllm_engine, monkeypatch):
    # _make_request is the single control-plane POST choke point; on a headless worker
    # (node_rank>0) it must no-op to None without issuing any HTTP request.
    def _boom(*a, **k):
        raise AssertionError("control-plane HTTP must not be called on a headless worker")

    monkeypatch.setattr(mod.requests, "post", _boom)
    vllm_engine.node_rank = 1
    assert vllm_engine._make_request("whatever", {}) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
