"""CPU unit tests for ``vime.backends.vllm_utils.arguments``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest

_real_vllm = _unit_stubs.real_module_available("vllm")
requires_vllm = pytest.mark.skipif(not _real_vllm, reason="requires real vllm install")

_unit_stubs.install_vllm_cli_stubs()

NUM_GPUS = 0


@pytest.fixture(scope="module")
def args_mod():
    from vime.backends.vllm_utils import arguments as mod  # noqa: PLC0415

    return mod


def _ns(**overrides):
    base = dict(
        vllm_data_parallel_size=1,
        vllm_pipeline_parallel_size=1,
        rollout_num_gpus_per_engine=4,
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
    # pp>1 and dp>1. Per-engine TP = gpus_per_engine // (pp * dp) is resolved at launch time.
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
def test_add_vllm_router_arguments_registers_vllm_prefix(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    assert "--vllm-router-ip" in flags
    assert "--vllm-router-port" in flags
    assert "--router-request-timeout-secs" in flags


@pytest.mark.unit
def test_add_vllm_router_arguments_dests(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "vllm_router_ip" in dests
    assert "vllm_router_port" in dests
    assert "router_request_timeout_secs" in dests


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
        ["--vllm-router-ip", "10.0.0.1", "--vllm-router-port", "8000", "--router-request-timeout-secs", "30"]
    )
    assert parsed.vllm_router_ip == "10.0.0.1"
    assert parsed.vllm_router_port == 8000
    assert parsed.router_request_timeout_secs == 30


@pytest.mark.unit
def test_add_vllm_router_arguments_defaults_to_consistent_hash(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    parsed, _ = parser.parse_known_args([])
    assert parsed.router_policy == "consistent_hash"


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
    assert "--vllm-weight-sync-packed" in flags


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
    """TP auto-compute must divide by DP: TP = gpus / (PP * DP)."""
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--rollout-num-gpus-per-engine", "8", "--vllm-data-parallel-size", "4"],
    )
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 2  # 8 / (1 * 4) = 2


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
