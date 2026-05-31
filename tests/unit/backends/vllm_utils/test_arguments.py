"""Unit tests for ``slime.backends.vllm_utils.arguments``."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def args_mod():
    from slime.backends.vllm_utils import arguments as mod  # noqa: PLC0415

    return mod


@pytest.mark.unit
def test_strip_unsupported_kwargs_on_312_real(args_mod):
    assert sys.version_info < (3, 13)
    out = args_mod._strip_unsupported_argparse_kwargs(
        {"type": int, "deprecated": True, "deprecated_aliases": ["x"], "help": "h"}
    )
    assert out == {"type": int, "help": "h"}


@pytest.mark.unit
def test_strip_unsupported_kwargs_passthrough_on_313(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "sys", SimpleNamespace(version_info=(3, 13, 0), argv=sys.argv))
    kw = {"type": int, "deprecated": True, "help": "h"}
    out = args_mod._strip_unsupported_argparse_kwargs(kw)
    assert out == kw


@pytest.mark.unit
def test_strip_unsupported_kwargs_noop_when_absent(args_mod):
    kwargs = {"type": int, "help": "h", "default": 0}
    out = args_mod._strip_unsupported_argparse_kwargs(kwargs)
    assert out == kwargs


@pytest.mark.unit
def test_wrapper_prefixes_long_flag_real_parser(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--gpu-memory-utilization", type=float, default=0.92)
    parsed, _ = parser.parse_known_args(["--vllm-gpu-memory-utilization", "0.5"])
    assert parsed.vllm_gpu_memory_utilization == 0.5


@pytest.mark.unit
def test_wrapper_prefixes_dest_real_parser(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--foo", dest="foo", type=int, default=0)
    parsed, _ = parser.parse_known_args(["--vllm-foo", "7"])
    assert parsed.vllm_foo == 7


@pytest.mark.unit
def test_wrapper_no_double_prefix(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--foo", dest="vllm_foo", type=int, default=0)
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "vllm_foo" in dests
    assert "vllm_vllm_foo" not in dests


@pytest.mark.unit
def test_wrapper_skips_dest_listed_in_SKIPPED_DESTS(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--tensor-parallel-size", type=int)
    flags = {s for a in parser._actions for s in a.option_strings}
    assert "--vllm-tensor-parallel-size" not in flags
    assert "--tensor-parallel-size" not in flags


@pytest.mark.unit
def test_SKIPPED_DESTS_orchestrator_parallel_dims(args_mod):
    """TP/multi-node dims are orchestrator-owned; PP/DP remain CLI-forwardable."""
    assert "tensor_parallel_size" in args_mod.SKIPPED_DESTS
    assert "pipeline_parallel_size" not in args_mod.SKIPPED_DESTS
    assert "data_parallel_size" not in args_mod.SKIPPED_DESTS
    assert "nnodes" in args_mod.SKIPPED_DESTS
    assert "node_rank" in args_mod.SKIPPED_DESTS
    assert "master_addr" in args_mod.SKIPPED_DESTS
    assert "master_port" in args_mod.SKIPPED_DESTS
    assert "data_parallel_backend" in args_mod.SKIPPED_DESTS
    assert "distributed_executor_backend" in args_mod.SKIPPED_DESTS


@pytest.mark.unit
def test_wrapper_strips_deprecated_kwargs_when_forwarding(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--foo", type=int, default=0, deprecated="oldname")
    parsed, _ = parser.parse_known_args(["--vllm-foo", "3"])
    assert parsed.vllm_foo == 3


@pytest.mark.unit
def test_detect_user_provided_value_form(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--foo", type=int, default=0)
    user, raw = args_mod._detect_user_provided_dests(parser, ["--foo", "5"])
    assert user == {"foo"}
    assert raw == {"foo": "5"}


@pytest.mark.unit
def test_detect_user_provided_equals_form(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bar", type=str, default="x")
    user, raw = args_mod._detect_user_provided_dests(parser, ["--bar=hello"])
    assert user == {"bar"}
    assert raw == {"bar": "hello"}


@pytest.mark.unit
def test_detect_user_provided_omitted(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baz", type=int, default=42)
    user, raw = args_mod._detect_user_provided_dests(parser, ["--other", "1"])
    assert "baz" not in user
    assert "baz" not in raw


@pytest.mark.unit
def test_detect_user_provided_ignores_unregistered_flags(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--known", type=int)
    user, raw = args_mod._detect_user_provided_dests(parser, ["--unknown", "v"])
    assert user == set()
    assert raw == {}


@pytest.mark.unit
def test_detect_user_provided_multiple(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--a", type=int, default=0)
    parser.add_argument("--b", type=str, default="")
    user, raw = args_mod._detect_user_provided_dests(parser, ["--a", "1", "--b=hello"])
    assert user == {"a", "b"}
    assert raw == {"a": "1", "b": "hello"}


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
    # validate_args intentionally does NOT set a global ``vllm_tp_size`` anymore: a global TP
    # (derived from the *global* rollout_num_gpus_per_engine) shadowed the per-engine value and
    # broke heterogeneous per-group engines (the 300s "3/4 clients joined" rendezvous hang). TP is
    # now derived per engine in vllm_engine._resolve_vllm_parallel_sizes (covered in
    # test_vllm_engine.py::test_resolve_parallel_sizes_is_per_engine_not_global).
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
    # The pp-divisibility check moved out of validate_args and into the per-engine resolver
    # (vllm_engine._resolve_vllm_parallel_sizes / compute_vllm_engine_topology), so validate_args
    # itself is now agnostic to it. The enforcement is covered in test_vllm_engine.py.
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


@pytest.mark.unit
def test_orchestration_dests_use_vllm_prefix(args_mod):
    assert "vllm_router_ip" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "vllm_router_port" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "router_request_timeout_secs" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "vllm_weight_transfer_timeout_sec" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "router_ip" not in args_mod._VIME_ORCHESTRATION_DESTS
    assert "router_port" not in args_mod._VIME_ORCHESTRATION_DESTS
    assert "vllm_router_request_timeout_secs" not in args_mod._VIME_ORCHESTRATION_DESTS


@pytest.mark.unit
def test_add_vllm_arguments_parses_weight_transfer_timeout(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod.AsyncEngineArgs, "add_cli_args", staticmethod(lambda parser: parser))
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_arguments(parser)
    default, _ = parser.parse_known_args([])
    assert default.vllm_weight_transfer_timeout_sec == 900.0
    parsed, _ = parser.parse_known_args(["--vllm-weight-transfer-timeout-sec", "123.5"])
    assert parsed.vllm_weight_transfer_timeout_sec == 123.5


def _realistic_add_vllm_arguments(parser):
    parser.add_argument("--vllm-gpu-memory-utilization", dest="vllm_gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--vllm-enforce-eager", dest="vllm_enforce_eager", action="store_true", default=False)
    parser.add_argument("--vllm-router-ip", dest="vllm_router_ip", type=str, default=None)
    parser.add_argument("--vllm-router-port", dest="vllm_router_port", type=int, default=None)
    parser.add_argument("--vllm-server-concurrency", dest="vllm_server_concurrency", type=int, default=512)
    parser.add_argument(
        "--vllm-weight-transfer-timeout-sec",
        dest="vllm_weight_transfer_timeout_sec",
        type=float,
        default=900.0,
    )
    return parser


@pytest.mark.unit
def test_action_table_caches(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "_VLLM_CLI_ACTION_TABLE_CACHE", None)
    monkeypatch.setattr(args_mod, "add_vllm_arguments", _realistic_add_vllm_arguments)
    t1 = args_mod.get_vllm_cli_action_table()
    t2 = args_mod.get_vllm_cli_action_table()
    assert t1 is t2


@pytest.mark.unit
def test_action_table_excludes_orchestration(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "_VLLM_CLI_ACTION_TABLE_CACHE", None)
    monkeypatch.setattr(args_mod, "add_vllm_arguments", _realistic_add_vllm_arguments)
    table = args_mod.get_vllm_cli_action_table()
    assert "vllm_gpu_memory_utilization" in table
    assert "vllm_enforce_eager" in table
    assert "vllm_router_ip" not in table
    assert "vllm_router_port" not in table
    assert "vllm_server_concurrency" not in table
    assert "vllm_weight_transfer_timeout_sec" not in table


@pytest.mark.unit
def test_action_table_flag_format(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "_VLLM_CLI_ACTION_TABLE_CACHE", None)
    monkeypatch.setattr(args_mod, "add_vllm_arguments", _realistic_add_vllm_arguments)
    table = args_mod.get_vllm_cli_action_table()
    flag, _action = table["vllm_gpu_memory_utilization"]
    assert flag == "--gpu-memory-utilization"


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
def test_parse_args_records_user_provided(args_mod, monkeypatch):
    def stub(parser):
        parser.add_argument("--vllm-foo", dest="vllm_foo", type=int, default=0)
        return parser

    monkeypatch.setattr(args_mod, "add_vllm_arguments", stub)
    monkeypatch.setattr(sys, "argv", ["train.py", "--vllm-foo", "7"])
    ns = args_mod.vllm_parse_args()
    assert "vllm_foo" in ns._vllm_user_provided
    assert ns._vllm_raw_values["vllm_foo"] == "7"


@pytest.mark.unit
def test_parse_args_default_attribute_set_even_without_register(args_mod, monkeypatch):
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(sys, "argv", ["train.py", "--rollout-num-gpus-per-engine", "8"])
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 8
