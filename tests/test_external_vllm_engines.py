import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vime.backends.vllm_utils.external import (
    ExternalEngineInfo,
    apply_external_engine_info_to_args,
    discover_external_engines,
    get_server_info,
    start_external_rollout_servers,
)
from vime.utils.http_utils import get_rollout_num_engines

NUM_GPUS = 0


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_discover_external_engines_reads_server_info(monkeypatch):
    def fake_get(url, timeout):
        assert timeout == 30.0
        assert url == "http://host1:10090/server_info?config_format=json"
        return _Response(
            {
                "tp_size": 4,
                "pp_size": 2,
                "pcp_size": 1,
                "dp_size": 1,
                "enable_expert_parallel": True,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", fake_get)

    infos = discover_external_engines(["host1:10090"])

    assert len(infos) == 1
    info = infos[0]
    assert info.url == "http://host1:10090"
    assert info.host == "host1"
    assert info.port == 10090
    assert info.worker_type == "regular"
    assert info.num_gpus == 8
    assert info.server_info["tp_size"] == 4
    assert info.server_info["pp_size"] == 2
    assert info.server_info["dp_size"] == 1
    assert info.server_info["enable_expert_parallel"] is True
    assert info.parallel_config == {
        "tp_size": 4,
        "pp_size": 2,
        "pcp_size": 1,
        "dp_size": 1,
        "enable_expert_parallel": True,
        "ep_size": 4,
    }


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "ec_connector, ec_role, kv_role, expected",
    [
        ("ECExampleConnector", "ec_producer", None, "encoder"),
        ("ECExampleConnector", "ec_consumer", "kv_producer", "prefill"),
        ("ECExampleConnector", "ec_consumer", "kv_consumer", "decode"),
        ("ECExampleConnector", "ec_both", None, "regular"),
        (None, "ec_producer", None, "regular"),
    ],
)
def test_discover_external_native_ec_roles(monkeypatch, nested, ec_connector, ec_role, kv_role, expected):
    config = {
        "ec_transfer_config": {"ec_connector": ec_connector, "ec_role": ec_role},
        "kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": kv_role},
        "parallel_config": {"tensor_parallel_size": 1},
    }
    payload = {"vllm_config": config} if nested else config
    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", lambda *args, **kwargs: _Response(payload))

    info = discover_external_engines(["host1:10090"])[0]

    assert info.worker_type == expected
    assert info.server_info["ec_transfer_config"] == config["ec_transfer_config"]


@pytest.mark.parametrize("worker_type", ["regular", "prefill", "decode"])
@pytest.mark.parametrize("prometheus_port", [None, 31000])
def test_start_external_rollout_servers_exposes_parallel_configs(monkeypatch, worker_type, prometheus_port):
    class FakeActor:
        init = Namespace(remote=lambda **kwargs: kwargs)

    class FakeActorClass:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            return FakeActor()

    ray = types.ModuleType("ray")
    ray.remote = lambda actor_class: FakeActorClass()
    vllm_engine = types.ModuleType("vime.backends.vllm_utils.vllm_engine")
    vllm_engine.VLLMEngine = object
    ray_utils = types.ModuleType("vime.ray.utils")
    ray_utils.add_default_ray_env_vars = lambda: {}
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "vime.backends.vllm_utils.vllm_engine", vllm_engine)
    monkeypatch.setitem(sys.modules, "vime.ray.utils", ray_utils)

    info = ExternalEngineInfo(
        url="http://host1:10090",
        host="host1",
        port=10090,
        worker_type=worker_type,
        num_gpus=8,
        server_info={
            "tp_size": 2,
            "pp_size": 2,
            "pcp_size": 1,
            "dp_size": 2,
            "enable_expert_parallel": True,
        },
    )
    args = Namespace(rollout_external_engine_infos=[info.to_dict()])

    servers, init_handles = start_external_rollout_servers(
        args,
        start_router=lambda *args, **kwargs: ("host1", 30000, prometheus_port),
    )

    assert servers["default"].engine_parallel_configs == [
        {
            "tp_size": 2,
            "pp_size": 2,
            "pcp_size": 1,
            "dp_size": 2,
            "enable_expert_parallel": True,
            "ep_size": 4,
        }
    ]
    assert servers["default"].prometheus_port == prometheus_port
    assert len(init_handles) == 1
    assert init_handles[0]["router_ip"] == "host1"
    assert init_handles[0]["router_port"] == 30000


def test_get_server_info_flattens_vllm_config(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://host1:10090/server_info?config_format=json"
        assert timeout == 30.0
        return _Response(
            {
                "vllm_config": {
                    "parallel_config": {"tensor_parallel_size": 2},
                    "kv_transfer_config": {
                        "kv_connector": "NixlConnector",
                        "kv_role": "kv_producer",
                    },
                    "weight_transfer_config": {"backend": "nccl"},
                },
                "vllm_env": {"VLLM_NIXL_SIDE_CHANNEL_PORT": 12090},
            }
        )

    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", fake_get)
    server_info = get_server_info("http://host1:10090")

    assert server_info["tensor_parallel_size"] == 2
    assert server_info["kv_transfer_config"]["kv_role"] == "kv_producer"
    assert server_info["weight_transfer_config"] == {"backend": "nccl"}
    assert server_info["disaggregation_mode"] == "prefill"
    assert server_info["disaggregation_bootstrap_port"] == 12090


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("address, expected_port", [(None, 22000), ("host1:23000", 23000), ("[::1]:23000", 23000)])
def test_external_mooncake_bootstrap_port(monkeypatch, multi, address, expected_port):
    connector = {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {} if address is None else {"bootstrap_server_address": address},
    }
    config = (
        {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"connectors": [connector]},
        }
        if multi
        else connector
    )
    payload = {
        "vllm_config": {"kv_transfer_config": config},
        "vllm_env": {"VLLM_NIXL_SIDE_CHANNEL_PORT": 5600, "VLLM_MOONCAKE_BOOTSTRAP_PORT": 22000},
    }
    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", lambda *args, **kwargs: _Response(payload))
    info = discover_external_engines(["host1:10090"])[0]
    assert info.worker_type == "prefill"
    assert info.disaggregation_bootstrap_port == expected_port
    assert info.server_info["kv_transfer_config"] == config


@pytest.mark.parametrize(
    "connector, role, multi, declared, expected",
    [
        ("MooncakeConnector", "kv_producer", True, None, "prefill"),
        ("NixlConnector", "kv_consumer", True, None, "decode"),
        ("SimpleCPUOffloadConnector", "kv_producer", False, None, "regular"),
        ("SimpleCPUOffloadConnector", "kv_consumer", False, None, "regular"),
        ("CustomConnector", "kv_producer", False, None, "regular"),
        ("CustomConnector", "kv_consumer", False, None, "regular"),
        ("MooncakeConnector", "kv_both", False, "prefill", "prefill"),
        ("NixlConnector", "kv_producer", True, "decode", "decode"),
    ],
)
def test_discovery_uses_known_pd_protocol_or_declared_mode(monkeypatch, connector, role, multi, declared, expected):
    config = {"kv_connector": connector, "kv_role": role}
    if multi:
        config = {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "connectors": [config, {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both"}]
            },
        }
    payload = {"vllm_config": {"kv_transfer_config": config}, "disaggregation_mode": declared}
    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", lambda *args, **kwargs: _Response(payload))
    info = discover_external_engines(["host1:10090"])[0]
    assert info.worker_type == expected
    assert info.server_info["kv_transfer_config"] == config


def test_apply_external_engine_info_handles_pd(monkeypatch):
    payloads = {
        "http://prefill:10090/server_info?config_format=json": {
            "tp_size": 2,
            "pp_size": 1,
            "pcp_size": 1,
            "dp_size": 1,
            "enable_expert_parallel": False,
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 12090,
        },
        "http://decode:10091/server_info?config_format=json": {
            "tp_size": 4,
            "pp_size": 1,
            "pcp_size": 1,
            "dp_size": 2,
            "enable_expert_parallel": True,
            "disaggregation_mode": "decode",
        },
    }

    def fake_get(url, timeout):
        return _Response(payloads[url])

    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["prefill:10090", "decode:10091"],
        rollout_num_gpus=None,
        rollout_num_gpus_per_engine=1,
        router_pd_disaggregation=False,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is False
    assert args.rollout_num_gpus == 10
    assert args.rollout_num_engines == 2
    assert get_rollout_num_engines(args) == 2
    assert [info["worker_type"] for info in args.rollout_external_engine_infos] == ["prefill", "decode"]
    assert [info["num_gpus"] for info in args.rollout_external_engine_infos] == [2, 8]
    assert [info["server_info"]["dp_size"] for info in args.rollout_external_engine_infos] == [1, 2]
    assert args.rollout_external_engine_infos[0]["disaggregation_bootstrap_port"] == 12090


def test_apply_external_engine_info_preserves_router_pd_flag(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://regular:10090/server_info?config_format=json"
        return _Response(
            {
                "tp_size": 2,
                "pp_size": 1,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("vime.backends.vllm_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["regular:10090"],
        router_pd_disaggregation=True,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is True
    assert args.rollout_num_gpus == 2
    assert args.rollout_num_engines == 1


def test_apply_external_engine_info_requires_addrs():
    args = Namespace(rollout_external_engine_addrs=None)

    with pytest.raises(ValueError, match="rollout-external-engine-addrs"):
        apply_external_engine_info_to_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
