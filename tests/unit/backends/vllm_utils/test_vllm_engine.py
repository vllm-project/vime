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
            raise RuntimeError(f"HTTP {self.status_code}")

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
def test_finish_weight_update_records_weight_version_after_success(vllm_engine, monkeypatch):
    def fake_post(endpoint: str, payload: dict, timeout: float):
        assert endpoint == "finish_weight_update"
        return _MockResponse(json_data={"done": True})

    monkeypatch.setattr(vllm_engine, "_post_json", fake_post)

    vllm_engine.finish_weight_update(weight_version="3")

    assert vllm_engine.get_weight_version() == "3"


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
def test_weight_transfer_http_timeout_reads_env(vllm_engine, monkeypatch):
    monkeypatch.setenv("SLIME_VLLM_WEIGHT_TRANSFER_UPDATE_TIMEOUT_SEC", "123.5")
    assert vllm_engine._weight_transfer_http_timeout() == 123.5


@pytest.mark.unit
def test_weight_transfer_http_timeout_fallback_to_legacy_env(vllm_engine, monkeypatch):
    monkeypatch.delenv("SLIME_VLLM_WEIGHT_TRANSFER_UPDATE_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("SLIME_VLLM_WEIGHT_TRANSFER_HTTP_TIMEOUT_SEC", "42")
    assert vllm_engine._weight_transfer_http_timeout() == 42.0


@pytest.mark.unit
def test_response_json_or_fallback_parses_dict():
    response = _MockResponse(json_data={"status": "ready"})
    assert mod._response_json_or_fallback(response) == {"status": "ready"}


@pytest.mark.unit
def test_response_json_or_fallback_non_dict_wrapped():
    response = _MockResponse()
    response.json = lambda: ["a", "b"]  # type: ignore[method-assign]
    assert mod._response_json_or_fallback(response) == {
        "ok": False,
        "error": "Response is not a dictionary",
        "data": ["a", "b"],
    }


@pytest.mark.unit
def test_response_json_or_fallback_invalid_json():
    response = _MockResponse(text="not-json")
    response.json = lambda: (_ for _ in ()).throw(ValueError("no json"))  # type: ignore[method-assign]
    assert mod._response_json_or_fallback(response) == {
        "ok": False,
        "error": "Invalid JSON response",
        "raw": "not-json",
    }


@pytest.mark.unit
def test_http_base_requires_init(vllm_args):
    from slime.backends.vllm_utils.vllm_engine import VLLMEngine

    engine = VLLMEngine(vllm_args, rank=0)
    with pytest.raises(RuntimeError, match="init\\(\\)"):
        engine._http_base()


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
