"""Unit tests for ``slime.backends.vllm_utils.vllm_engine``."""

from __future__ import annotations

import dataclasses
import json

import pytest
import requests
import torch

from slime.backends.vllm_utils import vllm_engine as mod


class _MockResponse:
    def __init__(self, *, json_data: dict | None = None, text: str = "", status_code: int = 200):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code

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
def test_to_local_gpu_id_without_cvd():
    assert mod._to_local_gpu_id(3) == 3


@pytest.mark.unit
def test_to_local_gpu_id_maps_physical_id(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4")
    assert mod._to_local_gpu_id(3) == 1


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
        return _MockResponse(json_data={"ok": True})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)

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
        return _MockResponse(json_data={"done": True})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)

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
        group_name="slime-pp_0",
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
        return _MockResponse(json_data={"status": "ok"})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)

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
        return _MockResponse(json_data={"ok": True})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)

    vllm_engine.start_weight_update()
    assert calls[0][2] == 123.5


@pytest.mark.unit
def test_init_weights_update_group_uses_config_timeout(vllm_engine, monkeypatch):
    vllm_engine.args.vllm_weight_transfer_timeout_sec = 123.5
    calls: list[tuple] = []

    def fake_post(endpoint: str, payload: dict, timeout: float):
        calls.append((endpoint, payload, timeout))
        return _MockResponse(json_data={"initialized": True})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)

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
    logged = mod._redact_cmd_for_log(cmd)
    assert "secret-token" not in logged
    assert "***" in logged


@pytest.mark.unit
def test_serialize_for_cli_primitives():
    assert mod._serialize_for_cli(42) == "42"
    assert mod._serialize_for_cli(True) == "True"
    assert mod._serialize_for_cli({"backend": "nccl"}) == json.dumps({"backend": "nccl"})


@pytest.mark.unit
def test_serialize_for_cli_dataclass():
    @dataclasses.dataclass
    class _Cfg:
        backend: str = "nccl"

    out = mod._serialize_for_cli(_Cfg())
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
        return _MockResponse(json_data={"initialized": True})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)
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
        "_post_json",
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
