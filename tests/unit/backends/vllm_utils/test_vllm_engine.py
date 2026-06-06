"""Unit tests for ``vime.backends.vllm_utils.vllm_engine``."""

from __future__ import annotations

import dataclasses
import json

import pytest
import requests
import torch

from vime.backends.vllm_utils import vllm_engine as mod


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
def test_format_v6_uri_wraps_ipv6():
    assert mod._format_v6_uri("2001:db8::1") == "[2001:db8::1]"


@pytest.mark.unit
def test_format_v6_uri_ipv4_unchanged():
    assert mod._format_v6_uri("10.0.0.1") == "10.0.0.1"


@pytest.mark.unit
def test_compute_vllm_engine_topology_single_node(vllm_args):
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 4
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_tp_size = 4
    topo = mod.compute_vllm_engine_topology(vllm_args, global_rank=0)
    assert topo.nnodes == 1
    assert topo.node_rank == 0
    assert topo.local_num_gpus == 4
    assert not topo.multi_node
    assert not topo.headless


@pytest.mark.unit
def test_compute_vllm_engine_topology_multi_node_ranks(vllm_args):
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 16
    vllm_args.vllm_pipeline_parallel_size = 2
    vllm_args.vllm_tp_size = 8
    topo0 = mod.compute_vllm_engine_topology(vllm_args, global_rank=0)
    topo1 = mod.compute_vllm_engine_topology(vllm_args, global_rank=1)
    assert topo0.nnodes == 2
    assert topo0.node_rank == 0
    assert topo1.node_rank == 1
    assert topo0.local_num_gpus == 8
    assert topo0.headless is False
    assert topo1.headless is True


@pytest.mark.unit
def test_append_distributed_flags_only_when_multi_node(vllm_args):
    cmd: list[str] = ["vllm", "serve"]
    single = mod.VllmEngineTopology(
        nnodes=1,
        node_rank=0,
        local_num_gpus=4,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )
    mod.append_vllm_distributed_launch_flags(cmd, single, ("10.0.0.1", 15000), vllm_args)
    assert cmd == ["vllm", "serve"]

    cmd_multi: list[str] = ["vllm", "serve"]
    multi = mod.VllmEngineTopology(
        nnodes=2,
        node_rank=1,
        local_num_gpus=8,
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
    )
    mod.append_vllm_distributed_launch_flags(cmd_multi, multi, ("10.0.0.2", 16000), vllm_args)
    assert "--nnodes" in cmd_multi
    assert "--node-rank" in cmd_multi
    assert "1" in cmd_multi
    assert "--headless" in cmd_multi
    assert "--master-addr" in cmd_multi
    assert "10.0.0.2" in cmd_multi
    assert cmd_multi[cmd_multi.index("--data-parallel-backend") + 1] == "mp"
    assert cmd_multi[cmd_multi.index("--distributed-executor-backend") + 1] == "mp"


@pytest.mark.unit
def test_parse_dist_init_addr_ipv6():
    host, port = mod.parse_dist_init_addr("[2001:db8::1]:15000")
    assert host == "2001:db8::1"
    assert port == 15000


@pytest.mark.unit
def test_build_vllm_subprocess_env_colocate(vllm_args, monkeypatch):
    vllm_args.colocate = True
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = mod.build_vllm_subprocess_env(
        {
            "args": vllm_args,
            "visible_devices": "0,1",
        }
    )
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" in env
    assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert "PYTHONPATH" in env


@pytest.mark.unit
def test_get_base_gpu_id_colocate(vllm_args):
    vllm_args.colocate = True
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 4
    assert mod.get_base_gpu_id(vllm_args, rank=1) == 4


@pytest.mark.unit
def test_start_weight_update_posts_four_phase_endpoint(vllm_engine, monkeypatch):
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict, timeout: float):
        calls.append((endpoint, payload, timeout))
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine.start_weight_update(is_checkpoint_format=True)

    assert result == {"ok": True}
    assert len(calls) == 1
    assert calls[0][0] == "start_weight_update"
    assert calls[0][1] == {"is_checkpoint_format": True}
    assert calls[0][2] == vllm_engine._weight_transfer_http_timeout()


@pytest.mark.unit
def test_finish_weight_update_posts_empty_body(vllm_engine, monkeypatch):
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict, timeout: float):
        calls.append((endpoint, payload, timeout))
        return {"done": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine.finish_weight_update()

    assert result == {"done": True}
    assert calls == [("finish_weight_update", {}, vllm_engine._weight_transfer_http_timeout())]


@pytest.mark.unit
def test_update_weights_from_tensor_posts_ipc_payload_and_records_version(vllm_engine, monkeypatch):
    posted: list[dict] = []
    monkeypatch.setattr(
        vllm_engine,
        "_post_vllm_update_weights_http",
        lambda payload: (posted.append(payload), {"ok": True})[1],
    )
    assert vllm_engine._weight_version is None

    vllm_engine.update_weights_from_tensor(
        names=["layer.0.weight"],
        dtype_names=["float32"],
        shapes=[[2, 2]],
        ipc_handles=[{"uuid-gpu0": ("rebuild_fn", (1, 2, 3))}],
        weight_version="42",
    )

    assert len(posted) == 1
    sent = posted[0]
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

    def fake_post_vllm_fail(payload: dict) -> dict:
        raise RuntimeError("simulated POST failure")

    monkeypatch.setattr(vllm_engine, "_post_vllm_update_weights_http", fake_post_vllm_fail)

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

    def fake_post_vllm(update_info: dict) -> dict:
        calls.append(update_info)
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_post_vllm_update_weights_http", fake_post_vllm)

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
def test_post_vllm_update_weights_http_wraps_update_info(vllm_engine, monkeypatch):
    seen: list[tuple] = []

    def fake_post(endpoint: str, payload: dict, timeout: float):
        seen.append((endpoint, payload, timeout))
        return {"status": "ok"}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    result = vllm_engine._post_vllm_update_weights_http({"names": ["w"], "packed": False})

    assert result == {"status": "ok"}
    assert seen[0][0] == "update_weights"
    assert seen[0][1] == {"update_info": {"names": ["w"], "packed": False}}


@pytest.mark.unit
def test_weight_transfer_http_timeout_reads_config(vllm_engine):
    vllm_engine.args.vllm_weight_transfer_timeout_sec = 123.5
    assert vllm_engine._weight_transfer_http_timeout() == 123.5


@pytest.mark.unit
def test_weight_transfer_http_timeout_uses_argument_default(vllm_engine):
    assert vllm_engine.args.vllm_weight_transfer_timeout_sec == 900.0
    assert vllm_engine._weight_transfer_http_timeout() == 900.0


@pytest.mark.unit
def test_start_weight_update_uses_config_timeout(vllm_engine, monkeypatch):
    vllm_engine.args.vllm_weight_transfer_timeout_sec = 123.5
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict, timeout: float):
        calls.append((endpoint, payload, timeout))
        return {"ok": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    vllm_engine.start_weight_update()
    assert calls[0][2] == 123.5


@pytest.mark.unit
def test_init_weights_update_group_uses_config_timeout(vllm_engine, monkeypatch):
    vllm_engine.args.vllm_weight_transfer_timeout_sec = 123.5
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict, timeout: float):
        calls.append((endpoint, payload, timeout))
        return {"initialized": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)

    vllm_engine.init_weights_update_group(
        "127.0.0.1",
        29500,
        rank_offset=1,
        world_size=4,
        group_name="unused",
        backend="nccl",
    )
    assert calls[0][2] == 123.5


@pytest.mark.unit
def test_response_json_parses_dict():
    response = _MockResponse(json_data={"status": "ready"})
    assert mod._response_json(response) == {"status": "ready"}


@pytest.mark.unit
def test_response_json_empty_body_returns_ok():
    response = _MockResponse(text="")
    assert mod._response_json(response) == {"ok": True}


@pytest.mark.unit
def test_response_json_invalid_json_raises():
    response = _MockResponse(text="not-json")
    response.json = lambda: (_ for _ in ()).throw(ValueError("no json"))  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="no json"):
        mod._response_json(response)


@pytest.mark.unit
def test_response_json_http_error_adds_response_text_note():
    response = _MockResponse(json_data={"error": "bad"}, text="server exploded", status_code=500)
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        mod._response_json(response)
    assert "response.text='server exploded'" in exc_info.value.__notes__


@pytest.mark.unit
def test_http_base_ipv6_host(vllm_engine):
    vllm_engine.server_host = "[2001:db8::1]"
    assert vllm_engine._http_base() == "http://[2001:db8::1]:8765"


@pytest.mark.unit
def test_redact_cmd_for_log_masks_hf_token():
    cmd = ["vllm", "serve", "model", "--hf-token", "secret-token", "--port", "8000"]
    logged = mod.redact_cmd_for_log(cmd)
    assert "secret-token" not in logged
    assert "***" in logged


@pytest.mark.unit
def test_serialize_for_cli_primitives():
    assert mod.serialize_for_cli(42) == "42"
    assert mod.serialize_for_cli(True) == "True"
    assert mod.serialize_for_cli({"backend": "nccl"}) == json.dumps({"backend": "nccl"})


@pytest.mark.unit
def test_serialize_for_cli_dataclass():
    @dataclasses.dataclass
    class _Cfg:
        backend: str = "nccl"

    out = mod.serialize_for_cli(_Cfg())
    assert json.loads(out) == {"backend": "nccl"}


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
def test_resume_memory_occupation_skips_when_sleep_disabled(vllm_engine):
    vllm_engine.args.vllm_enable_sleep_mode = False
    assert vllm_engine.resume_memory_occupation() == {"ok": True, "sleep_mode": False}


@pytest.mark.unit
def test_init_weights_update_group_retries_then_succeeds(vllm_engine, monkeypatch):
    attempts = {"n": 0}

    def fake_post(endpoint: str, payload: dict, timeout: float):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise requests.ConnectionError("transient")
        return {"initialized": True}

    monkeypatch.setattr(vllm_engine, "_make_request", fake_post)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    result = vllm_engine.init_weights_update_group(
        "127.0.0.1",
        29500,
        rank_offset=1,
        world_size=4,
        group_name="unused",
        backend="nccl",
    )

    assert result == {"initialized": True}
    assert attempts["n"] == 2


@pytest.mark.unit
def test_init_weights_update_group_raises_after_three_failures(vllm_engine, monkeypatch):
    monkeypatch.setattr(
        vllm_engine,
        "_make_request",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="init_weight_transfer_engine failed"):
        vllm_engine.init_weights_update_group(
            "127.0.0.1",
            29500,
            rank_offset=1,
            world_size=4,
            group_name="g",
            backend="nccl",
        )


def _stub_server_info(monkeypatch, parallel_config: dict) -> None:
    def fake_get(url, *, params=None, timeout=30):
        return _MockResponse(json_data={"vllm_config": {"parallel_config": parallel_config}})

    monkeypatch.setattr(mod.requests, "get", fake_get)


@pytest.mark.unit
def test_sanity_check_external_server_args_passes_on_match(vllm_engine, monkeypatch):
    vllm_engine._server_args = {"tp_size": 2, "pp_size": 1, "dp_size": 1, "nnodes": 1}
    _stub_server_info(
        monkeypatch,
        {"tensor_parallel_size": 2, "pipeline_parallel_size": 1, "data_parallel_size": 1, "nnodes": 1},
    )
    # All reported fields match the per-engine expectation → no raise.
    vllm_engine._sanity_check_external_server_args()


@pytest.mark.unit
def test_sanity_check_external_server_args_raises_on_tp_mismatch(vllm_engine, monkeypatch):
    # Expect a tp=2 engine but the external server reports tp=1 → fail fast (the bug class
    # that used to only warn and then hang the weight-sync rendezvous 300s later).
    vllm_engine._server_args = {"tp_size": 2, "pp_size": 1, "dp_size": 1, "nnodes": 1}
    _stub_server_info(
        monkeypatch,
        {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1, "nnodes": 1},
    )
    with pytest.raises(AssertionError, match="tp_size"):
        vllm_engine._sanity_check_external_server_args()


@pytest.mark.unit
def test_sanity_check_external_server_args_skips_unreported_field(vllm_engine, monkeypatch):
    # vLLM /server_info may not surface ``nnodes``: an unreported (None) field is skipped,
    # not treated as a mismatch — so a single-node external engine doesn't false-fail.
    vllm_engine._server_args = {"tp_size": 1, "pp_size": 1, "dp_size": 1, "nnodes": 2}
    _stub_server_info(
        monkeypatch,
        {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1},
    )
    vllm_engine._sanity_check_external_server_args()


@pytest.mark.unit
def test_sanity_check_external_server_args_raises_when_parallel_config_missing(vllm_engine, monkeypatch):
    vllm_engine._server_args = {"tp_size": 1, "pp_size": 1, "dp_size": 1, "nnodes": 1}
    _stub_server_info(monkeypatch, {})
    with pytest.raises(RuntimeError, match="missing vllm_config.parallel_config"):
        vllm_engine._sanity_check_external_server_args()


@pytest.mark.unit
def test_resolve_parallel_sizes_is_per_engine_not_global(vllm_args):
    # The global flag is 1, but THIS engine has 2 GPUs → tp must be 2 (per-engine), not 1.
    # A stale global vllm_tp_size must NOT shadow the per-engine value.
    vllm_args.rollout_num_gpus_per_engine = 1
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_tp_size = 1  # stale global; must be ignored now
    tp, pp = mod._resolve_vllm_parallel_sizes(vllm_args, gpus_per_engine=2)
    assert (tp, pp) == (2, 1)


@pytest.mark.unit
def test_compute_topology_heterogeneous_per_group_tp(vllm_args):
    # Reproduces the rendezvous bug's root: global=1 but a per-group engine uses 2 GPUs.
    vllm_args.num_gpus_per_node = 8
    vllm_args.rollout_num_gpus_per_engine = 1
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_tp_size = 1  # stale global
    topo = mod.compute_vllm_engine_topology(vllm_args, global_rank=0, num_gpus_per_engine=2)
    assert topo.tensor_parallel_size == 2
    assert topo.nnodes == 1


@pytest.mark.unit
def test_resolve_parallel_sizes_rejects_dp_gt_1(vllm_args):
    vllm_args.vllm_pipeline_parallel_size = 1
    vllm_args.vllm_data_parallel_size = 2
    vllm_args.vllm_dp_size = 2
    with pytest.raises(NotImplementedError, match="data parallelism"):
        mod._resolve_vllm_parallel_sizes(vllm_args, gpus_per_engine=4)


@pytest.mark.unit
def test_make_request_short_circuits_on_headless(vllm_engine, monkeypatch):
    # _make_request is the single control-plane POST choke point; on a headless worker
    # (node_rank>0) it must no-op to None without issuing any HTTP request.
    def _boom(*a, **k):
        raise AssertionError("control-plane HTTP must not be called on a headless worker")

    monkeypatch.setattr(mod.requests, "post", _boom)
    vllm_engine.node_rank = 1
    assert vllm_engine._make_request("whatever", {}, timeout=1) is None


@pytest.mark.unit
def test_control_plane_methods_noop_on_headless_worker(vllm_engine, monkeypatch):
    """node_rank>0 (headless) workers own no HTTP server; every control-plane method must
    no-op (return None) without issuing an HTTP request."""

    def _boom(*a, **k):
        raise AssertionError("control-plane HTTP must not be called on a headless worker")

    monkeypatch.setattr(mod.requests, "post", _boom)
    monkeypatch.setattr(mod.requests, "get", _boom)
    vllm_engine.node_rank = 1
    vllm_engine.args.vllm_enable_sleep_mode = True

    assert vllm_engine.init_weight_transfer_engine({"init_info": {}}) is None
    assert vllm_engine.start_weight_update() is None
    assert vllm_engine.finish_weight_update() is None
    assert vllm_engine.init_weights_update_group("addr", 1, 0, 4, "g", "nccl") is None
    assert vllm_engine.update_weights_from_distributed(["w"], [torch.float32], [[1]], "g") is None
    assert vllm_engine.release_memory_occupation() is None
    assert vllm_engine.resume_memory_occupation() is None
