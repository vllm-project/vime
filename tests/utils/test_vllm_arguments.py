"""CPU unit tests for ``vime.backends.vllm_utils.arguments``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest

_real_vllm = _unit_stubs.real_module_available("vllm")
requires_vllm = pytest.mark.skipif(not _real_vllm, reason="requires real vllm install")

_unit_stubs.install_vllm_cli_stubs()

NUM_GPUS = 0


@pytest.mark.parametrize("module", [ModuleType("unit_stub"), SimpleNamespace(), MagicMock(), None])
def test_real_module_available_rejects_test_stubs(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "unit_stub", module)
    assert not _unit_stubs.real_module_available("unit_stub")


def test_real_module_available_preserves_installed_modules(monkeypatch):
    assert _unit_stubs.real_module_available("json")
    monkeypatch.delitem(sys.modules, "json")
    assert _unit_stubs.real_module_available("json")
    assert not _unit_stubs.real_module_available("_vime_missing_unit_dependency")


@requires_vllm
@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], None), (["--vllm-log-error-stack"], True), (["--no-vllm-log-error-stack"], False)],
)
def test_frontend_defaults_are_resolved_by_the_server(args_mod, monkeypatch, flags, expected):
    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser()
    args_mod.add_vllm_arguments(parser)
    assert parser.parse_args(flags).vllm_log_error_stack is expected


@requires_vllm
@pytest.mark.parametrize(
    ("flags", "timeout", "abort"),
    [
        ([], 300, True),
        (["--vllm-watchdog-config", '{"timeout":3600,"abort_on_timeout":true}'], 3600, True),
        (["--vllm-watchdog-config", '{"abort_on_timeout":false}'], 300, False),
        (["--vllm-watchdog-config", '{"timeout":3600}'], 3600, False),
    ],
)
def test_watchdog_defaults_and_explicit_native_config(args_mod, monkeypatch, flags, timeout, abort):
    from vllm.config import WatchdogConfig
    from vime.backends.vllm_utils.runtime import _json_default

    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser()
    args_mod.add_vllm_arguments(parser)
    wire = json.loads(json.dumps(parser.parse_args(flags).vllm_watchdog_config, default=_json_default))
    config = WatchdogConfig(**wire)
    assert config.timeout == timeout
    assert config.abort_on_timeout is abort


def test_python_executable_is_a_launcher_option(args_mod, monkeypatch):
    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser()
    args_mod.add_vllm_arguments(parser)
    assert parser.parse_args([]).vllm_python_executable is None
    executable = "/path with spaces/serve-python.sh"
    assert parser.parse_args(["--vllm-python-executable", executable]).vllm_python_executable == executable


@requires_vllm
@pytest.mark.parametrize("engine_id", ["omitted", None, "explicit-engine"])
def test_kv_transfer_cli_retains_only_explicit_validated_fields(args_mod, monkeypatch, engine_id):
    from dataclasses import is_dataclass

    from pydantic import TypeAdapter
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs

    _patch_device_config(monkeypatch)
    config = {"kv_connector": "MooncakeConnector", "kv_role": "kv_both", "kv_port": "12000"}
    if engine_id != "omitted":
        config["engine_id"] = engine_id
    native_parser = argparse.ArgumentParser()
    AsyncEngineArgs.add_cli_args(native_parser)
    native = native_parser.parse_args(["--kv-transfer-config", json.dumps(config)]).kv_transfer_config
    assert is_dataclass(native)
    assert native.kv_port == 12000
    assert isinstance(native.engine_id, str)
    parser = argparse.ArgumentParser()
    args_mod.add_vllm_arguments(parser)
    forwarded = parser.parse_args(["--vllm-kv-transfer-config", json.dumps(config)]).vllm_kv_transfer_config
    expected = {"kv_connector": "MooncakeConnector", "kv_role": "kv_both", "kv_port": 12000}
    if engine_id == "explicit-engine":
        expected["engine_id"] = engine_id
    assert forwarded == expected
    first = TypeAdapter(KVTransferConfig).validate_python(forwarded)
    second = TypeAdapter(KVTransferConfig).validate_python(forwarded)
    if engine_id == "explicit-engine":
        assert native.engine_id == first.engine_id == second.engine_id == engine_id
    else:
        assert first.engine_id != second.engine_id


@pytest.fixture(scope="module")
def args_mod():
    from vime.backends.vllm_utils import arguments as mod  # noqa: PLC0415

    return mod


def _ns(**overrides):
    base = dict(
        vllm_data_parallel_size=1,
        vllm_pipeline_parallel_size=1,
        vllm_prefill_context_parallel_size=1,
        rollout_num_gpus_per_engine=4,
        rollout_top_k=-1,
        rollout_top_p=1.0,
        vllm_router_ip=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.unit
def test_validate_args_pp1(args_mod):
    ns = _ns()
    args_mod.validate_args(ns)
    assert ns.vllm_pp_size == 1
    assert ns.vllm_dp_size == 1
    assert not hasattr(ns, "vllm_tp_size")


@pytest.mark.unit
def test_validate_args_records_pp_dp_but_no_global_tp(args_mod):
    # validate_args records pp/dp on the namespace but must not precompute a global TP, even when
    # pp>1 and dp>1. Per-engine TP = gpus_per_engine // (pp * pcp * dp) is resolved at launch time.
    ns = _ns(vllm_pipeline_parallel_size=2, vllm_data_parallel_size=2)
    args_mod.validate_args(ns)
    assert ns.vllm_pp_size == 2
    assert ns.vllm_dp_size == 2
    assert not hasattr(ns, "vllm_tp_size")


@pytest.mark.unit
def test_validate_args_no_longer_raises_on_pp_indivisible(args_mod):
    ns = _ns(vllm_pipeline_parallel_size=3, rollout_num_gpus_per_engine=4)
    args_mod.validate_args(ns)  # must not raise
    assert ns.vllm_pp_size == 3


@pytest.mark.unit
def test_validate_args_router_ipv6_wrapped(args_mod):
    ns = _ns(vllm_router_ip="::1")
    args_mod.validate_args(ns)
    assert ns.vllm_router_ip == "[::1]"


@pytest.mark.unit
def test_validate_args_router_ipv6_already_wrapped_unchanged(args_mod):
    ns = _ns(vllm_router_ip="[::1]")
    args_mod.validate_args(ns)
    assert ns.vllm_router_ip == "[::1]"


@pytest.mark.unit
def test_validate_args_router_ipv4_unchanged(args_mod):
    ns = _ns(vllm_router_ip="127.0.0.1")
    args_mod.validate_args(ns)
    assert ns.vllm_router_ip == "127.0.0.1"


@pytest.mark.unit
def test_validate_args_router_none_noop(args_mod):
    ns = _ns(vllm_router_ip=None)
    args_mod.validate_args(ns)
    assert ns.vllm_router_ip is None


@pytest.mark.unit
def test_validate_args_accepts_unbounded_top_p_replay(args_mod):
    args_mod.validate_args(_ns(rollout_top_p=0.95))


@pytest.mark.unit
def test_validate_args_accepts_bounded_top_p_replay(args_mod):
    args_mod.validate_args(_ns(rollout_top_p=0.95, rollout_top_k=20))


@pytest.mark.unit
def test_add_vllm_router_arguments_registers_vllm_prefix(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    assert "--vllm-router-ip" in flags
    assert "--vllm-router-port" in flags
    assert "--vllm-router-request-timeout-secs" in flags


@pytest.mark.unit
def test_add_vllm_router_arguments_dests(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "vllm_router_ip" in dests
    assert "vllm_router_port" in dests
    assert "vllm_router_request_timeout_secs" in dests


@pytest.mark.unit
def test_add_vllm_router_arguments_no_unprefixed_names(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "--router-ip" not in flags
    assert "--router-port" not in flags
    assert "router_ip" not in dests
    assert "router_port" not in dests


@pytest.mark.unit
def test_add_vllm_router_arguments_parses_real_values(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    parsed, _ = parser.parse_known_args(
        ["--vllm-router-ip", "10.0.0.1", "--vllm-router-port", "8000", "--vllm-router-request-timeout-secs", "30"]
    )
    assert parsed.vllm_router_ip == "10.0.0.1"
    assert parsed.vllm_router_port == 8000
    assert parsed.vllm_router_request_timeout_secs == 30


@pytest.mark.unit
def test_add_vllm_router_arguments_defaults_to_cache_aware(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    parsed, _ = parser.parse_known_args([])
    assert parsed.router_policy == "cache_aware"
    assert parsed.router_kv_connector == "mooncake"
    parsed = parser.parse_args(["--router-kv-connector", "nixl"])
    assert parsed.router_kv_connector == "nixl"


@pytest.mark.unit
@pytest.mark.parametrize("flags, expected", [([], "warning"), (["--router-log-level", "debug"], "debug")])
@pytest.mark.parametrize("has_pd_disaggregation", [False, True])
def test_router_log_level_survives_launch(args_mod, monkeypatch, flags, expected, has_pd_disaggregation):
    from vllm_router.router_args import RouterArgs

    from vime.backends.vllm_utils import deployment

    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    args = parser.parse_args(flags)
    assert args.router_log_level == expected
    router_args = SimpleNamespace(
        log_level=args.router_log_level, kv_connector="mooncake", intra_node_data_parallel_size=1
    )
    monkeypatch.setattr(RouterArgs, "from_cli_args", lambda *args, **kwargs: router_args)
    launches = []

    def make_process(*, target, args):
        launches.append(args[0])
        return SimpleNamespace(start=lambda: None, is_alive=lambda: True)

    monkeypatch.setattr(deployment, "find_available_port", lambda port: port)
    monkeypatch.setattr(deployment.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(deployment.multiprocessing, "Process", make_process)
    monkeypatch.setattr(deployment, "get_host_info", lambda: (None, "127.0.0.1"))
    args.vllm_router_ip = None
    args.vllm_router_port = 30000
    args.vllm_data_parallel_size = 4
    deployment._start_router(args, has_pd_disaggregation=has_pd_disaggregation)
    assert launches[0].log_level == expected
    assert launches[0].intra_node_data_parallel_size == (4 if has_pd_disaggregation else 1)


@pytest.mark.unit
def test_add_vllm_arguments_overrides_router_balance_threshold_defaults(args_mod, monkeypatch):
    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_arguments(parser)
    parsed, _ = parser.parse_known_args([])
    assert parsed.router_balance_abs_threshold == 10
    assert parsed.router_balance_rel_threshold == 1.2


def _patch_device_config(monkeypatch):
    """Patch DeviceConfig.__post_init__ to avoid GPU device detection on CPU CI."""
    try:
        from vllm.config.device import DeviceConfig

        monkeypatch.setattr(DeviceConfig, "__post_init__", lambda self: setattr(self, "device_type", "cpu"))
    except ImportError:
        pass


@pytest.mark.unit
@requires_vllm
def test_add_vllm_arguments_prefixes_regular_engine_flags(args_mod, monkeypatch):
    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    assert "--vllm-server-concurrency" in flags
    assert "--vllm-tool-call-parser" in flags
    assert "--vllm-weight-sync-packed" not in flags
    assert "--no-vllm-weight-sync-packed" not in flags


@pytest.mark.unit
def test_add_vllm_arguments_skips_orchestrator_owned_fields(args_mod, monkeypatch):
    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "--vllm-seed" not in flags
    assert "--vllm-host" not in flags
    assert "--vllm-master-addr" not in flags
    assert "--vllm-tensor-parallel-size" not in flags
    assert "seed" not in dests
    assert "host" not in dests
    assert "master_addr" not in dests
    assert "tensor_parallel_size" not in dests


@pytest.mark.unit
@requires_vllm
def test_add_vllm_arguments_parses_prefixed_engine_values(args_mod, monkeypatch):
    _patch_device_config(monkeypatch)
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_arguments(parser)
    parsed, _ = parser.parse_known_args(["--vllm-server-concurrency", "128", "--vllm-tool-call-parser", "qwen3_coder"])
    assert parsed.vllm_server_concurrency == 128
    assert parsed.vllm_tool_call_parser == "qwen3_coder"


@pytest.mark.unit
def test_parse_args_tp_default_no_pp(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(sys, "argv", ["train.py", "--rollout-num-gpus-per-engine", "4"])
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 4


@pytest.mark.unit
def test_parse_args_tp_default_with_pp(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--rollout-num-gpus-per-engine", "4", "--vllm-pipeline-parallel-size", "2"],
    )
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 2


@pytest.mark.unit
def test_parse_args_default_attribute_set_even_without_register(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(sys, "argv", ["train.py", "--rollout-num-gpus-per-engine", "8"])
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 8


@pytest.mark.unit
def test_parse_args_tp_default_with_dp(args_mod, monkeypatch):
    """TP auto-compute must divide by DP: TP = gpus / (PP * PCP * DP)."""
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--rollout-num-gpus-per-engine", "8", "--vllm-data-parallel-size", "4"],
    )
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 2  # 8 / (1 * 4) = 2


@pytest.mark.unit
def test_parse_args_tp_default_with_pcp_and_dp(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--rollout-num-gpus-per-engine",
            "8",
            "--vllm-prefill-context-parallel-size",
            "2",
            "--vllm-data-parallel-size",
            "2",
        ],
    )

    ns = args_mod.vllm_parse_args()

    assert ns.vllm_tensor_parallel_size == 2


@pytest.mark.unit
def test_parse_args_tp_default_with_pp_and_dp(args_mod, monkeypatch):
    """TP = gpus / (PP * DP) when both PP and DP are set."""
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--rollout-num-gpus-per-engine",
            "8",
            "--vllm-pipeline-parallel-size",
            "2",
            "--vllm-data-parallel-size",
            "2",
        ],
    )
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 2  # 8 / (2 * 2) = 2


@pytest.mark.unit
def test_parse_args_tp_default_dp1_unchanged(args_mod, monkeypatch):
    """DP=1 (default) must not change existing TP behavior."""
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--rollout-num-gpus-per-engine", "4", "--vllm-data-parallel-size", "1"],
    )
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 4  # 4 / (1 * 1) = 4


@pytest.mark.unit
def test_validate_args_rejects_conflicting_rollout_external_and_vllm_config(args_mod):
    ns = _ns(rollout_external=True, vllm_config="/tmp/vllm.yaml")
    with pytest.raises(AssertionError, match="vllm_config cannot be set"):
        args_mod.validate_args(ns)


@pytest.mark.unit
def test_validate_args_rejects_conflicting_prefill_and_vllm_config(args_mod):
    ns = _ns(prefill_num_servers=2, vllm_config="/tmp/vllm.yaml")
    with pytest.raises(AssertionError, match="mutually exclusive"):
        args_mod.validate_args(ns)


@pytest.mark.unit
def test_validate_args_rejects_prefill_and_rollout_external(args_mod):
    ns = _ns(prefill_num_servers=2, rollout_external=True)
    with pytest.raises(AssertionError, match="cannot be set"):
        args_mod.validate_args(ns)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
