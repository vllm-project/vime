"""Unit tests for changes in PR #10 to slime/backends/vllm_utils/arguments.py.

Test philosophy: minimize mocks. Use real `argparse.ArgumentParser`, real
`SimpleNamespace`, real `vllm_router.RouterArgs.add_cli_args`. Mock only what
genuinely cannot run in CPU-only test environment:
  * `vllm.engine.arg_utils.AsyncEngineArgs.add_cli_args` — instantiates a
    `VllmConfig` which requires CUDA (we don't have GPU in pytest container).
  * `sys.version_info` — to exercise both Python <3.13 and >=3.13 branches
    of `_strip_unsupported_argparse_kwargs` from the same test run.
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def args_mod():
    from slime.backends.vllm_utils import arguments as mod  # noqa: PLC0415
    return mod


# ============================================================================
# Fix #6 (issue-level): Python 3.12 argparse compat
# ============================================================================


@pytest.mark.unit
def test_strip_unsupported_kwargs_on_312_real(args_mod):
    """On the real test interpreter (3.12), `deprecated` should be stripped."""
    assert sys.version_info < (3, 13), "this test assumes the test interpreter is <3.13"
    out = args_mod._strip_unsupported_argparse_kwargs(
        {"type": int, "deprecated": True, "deprecated_aliases": ["x"], "help": "h"}
    )
    assert out == {"type": int, "help": "h"}


@pytest.mark.unit
def test_strip_unsupported_kwargs_passthrough_on_313(args_mod, monkeypatch):
    """On Python 3.13+, argparse accepts these natively — must pass through.

    We monkeypatch `sys.version_info` because we can't run two interpreters
    in one test session; the assertion is on the function's branching logic.
    """
    monkeypatch.setattr(
        args_mod, "sys", SimpleNamespace(version_info=(3, 13, 0), argv=sys.argv)
    )
    kw = {"type": int, "deprecated": True, "help": "h"}
    out = args_mod._strip_unsupported_argparse_kwargs(kw)
    assert out == kw


@pytest.mark.unit
def test_strip_unsupported_kwargs_noop_when_absent(args_mod):
    """No-op on well-formed kwargs."""
    kwargs = {"type": int, "help": "h", "default": 0}
    out = args_mod._strip_unsupported_argparse_kwargs(kwargs)
    assert out == kwargs


# ============================================================================
# `_make_add_argument_wrapper`: prefix flags + dest, skip SKIPPED_DESTS,
# strip unsupported kwargs. Test against a *real* argparse.ArgumentParser.
# ============================================================================


@pytest.mark.unit
def test_wrapper_prefixes_long_flag_real_parser(args_mod):
    """The wrapper rewrites `--foo-bar` to `--vllm-foo-bar` when forwarding to
    a real argparse parser."""
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
    """Caller-supplied `dest='vllm_foo'` must not become `vllm_vllm_foo`."""
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--foo", dest="vllm_foo", type=int, default=0)
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "vllm_foo" in dests
    assert "vllm_vllm_foo" not in dests


@pytest.mark.unit
def test_wrapper_skips_dest_listed_in_SKIPPED_DESTS(args_mod):
    """`tensor_parallel_size` is in SKIPPED_DESTS — wrapper must not register."""
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    wrap("--tensor-parallel-size", type=int)
    flags = {s for a in parser._actions for s in a.option_strings}
    assert "--vllm-tensor-parallel-size" not in flags
    assert "--tensor-parallel-size" not in flags


@pytest.mark.unit
def test_SKIPPED_DESTS_post_pr10_unskips_pp_and_dp(args_mod):
    """PR #10 unskips pipeline_parallel_size and data_parallel_size; only
    tensor_parallel_size remains in SKIPPED_DESTS (vime orchestrator owns it).
    """
    assert "tensor_parallel_size" in args_mod.SKIPPED_DESTS
    assert "pipeline_parallel_size" not in args_mod.SKIPPED_DESTS
    assert "data_parallel_size" not in args_mod.SKIPPED_DESTS


@pytest.mark.unit
def test_wrapper_strips_deprecated_kwargs_when_forwarding(args_mod):
    """End-to-end: an argparse parser receiving `deprecated=...` from vllm
    must not raise on Python <3.13 (the wrapper strips it first)."""
    parser = argparse.ArgumentParser(add_help=False)
    wrap = args_mod._make_add_argument_wrapper(parser.add_argument)
    # On <3.13 this would raise TypeError without the strip.
    wrap("--foo", type=int, default=0, deprecated="oldname")
    # Verify the action was registered (no exception, dest present)
    parsed, _ = parser.parse_known_args(["--vllm-foo", "3"])
    assert parsed.vllm_foo == 3


# ============================================================================
# `_detect_user_provided_dests`: parse argv against a real argparse parser.
# ============================================================================


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
    user, raw = args_mod._detect_user_provided_dests(
        parser, ["--a", "1", "--b=hello"]
    )
    assert user == {"a", "b"}
    assert raw == {"a": "1", "b": "hello"}


# ============================================================================
# `validate_args`: mirrors slime — TP/PP/DP aliases + TP derivation +
# router_ip IPv6 wrap. No mocks; uses real SimpleNamespace.
# ============================================================================


def _ns(**overrides):
    base = dict(
        vllm_data_parallel_size=1,
        vllm_pipeline_parallel_size=1,
        rollout_num_gpus_per_engine=4,
        router_ip=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.unit
def test_validate_args_pp1(args_mod):
    ns = _ns()
    args_mod.validate_args(ns)
    assert ns.vllm_pp_size == 1
    assert ns.vllm_dp_size == 1
    assert ns.vllm_tp_size == 4


@pytest.mark.unit
def test_validate_args_pp2_dp2_derives_tp(args_mod):
    ns = _ns(vllm_pipeline_parallel_size=2, vllm_data_parallel_size=2)
    args_mod.validate_args(ns)
    assert ns.vllm_pp_size == 2
    assert ns.vllm_dp_size == 2
    assert ns.vllm_tp_size == 2


@pytest.mark.unit
def test_validate_args_pp_indivisible_asserts(args_mod):
    ns = _ns(vllm_pipeline_parallel_size=3, rollout_num_gpus_per_engine=4)
    with pytest.raises(AssertionError, match="divisible"):
        args_mod.validate_args(ns)


@pytest.mark.unit
def test_validate_args_router_ipv6_wrapped(args_mod):
    ns = _ns(router_ip="::1")
    args_mod.validate_args(ns)
    assert ns.router_ip == "[::1]"


@pytest.mark.unit
def test_validate_args_router_ipv6_already_wrapped_unchanged(args_mod):
    ns = _ns(router_ip="[::1]")
    args_mod.validate_args(ns)
    assert ns.router_ip == "[::1]"


@pytest.mark.unit
def test_validate_args_router_ipv4_unchanged(args_mod):
    ns = _ns(router_ip="127.0.0.1")
    args_mod.validate_args(ns)
    assert ns.router_ip == "127.0.0.1"


@pytest.mark.unit
def test_validate_args_router_none_noop(args_mod):
    ns = _ns(router_ip=None)
    args_mod.validate_args(ns)
    assert ns.router_ip is None


# ============================================================================
# `add_vllm_router_arguments`: registers the THREE renamed flags
# (--router-ip / --router-port / --router-request-timeout-secs).
# No mocks — direct real argparse exercise.
# ============================================================================


@pytest.mark.unit
def test_add_vllm_router_arguments_registers_router_prefix(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    assert "--router-ip" in flags
    assert "--router-port" in flags
    assert "--router-request-timeout-secs" in flags


@pytest.mark.unit
def test_add_vllm_router_arguments_dests(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "router_ip" in dests
    assert "router_port" in dests
    assert "router_request_timeout_secs" in dests


@pytest.mark.unit
def test_add_vllm_router_arguments_old_names_removed(args_mod):
    """Regression guard: the old `--vllm-router-*` names must not exist."""
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    flags = {s for a in parser._actions for s in a.option_strings}
    dests = {a.dest for a in parser._actions if a.option_strings}
    assert "--vllm-router-ip" not in flags
    assert "--vllm-router-port" not in flags
    assert "vllm_router_ip" not in dests
    assert "vllm_router_port" not in dests


@pytest.mark.unit
def test_add_vllm_router_arguments_parses_real_values(args_mod):
    parser = argparse.ArgumentParser(add_help=False)
    args_mod.add_vllm_router_arguments(parser)
    parsed, _ = parser.parse_known_args(
        ["--router-ip", "10.0.0.1", "--router-port", "8000",
         "--router-request-timeout-secs", "30"]
    )
    assert parsed.router_ip == "10.0.0.1"
    assert parsed.router_port == 8000
    assert parsed.router_request_timeout_secs == 30


# ============================================================================
# `_VIME_ORCHESTRATION_DESTS`: renamed dests sit here, old names purged.
# ============================================================================


@pytest.mark.unit
def test_orchestration_dests_use_new_names(args_mod):
    assert "router_ip" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "router_port" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "router_request_timeout_secs" in args_mod._VIME_ORCHESTRATION_DESTS
    assert "vllm_router_ip" not in args_mod._VIME_ORCHESTRATION_DESTS
    assert "vllm_router_port" not in args_mod._VIME_ORCHESTRATION_DESTS


# ============================================================================
# `get_vllm_cli_action_table` caching and orchestration-dest exclusion.
# We stub `add_vllm_arguments` with a real (but minimal) parser registration —
# *not* a MagicMock. The stub is necessary only because the real function
# calls `AsyncEngineArgs.add_cli_args` which needs a CUDA device.
# ============================================================================


def _realistic_add_vllm_arguments(parser):
    """Minimal stand-in for `add_vllm_arguments` that uses real argparse calls
    instead of `MagicMock`. Registers a representative mix of dests for the
    test cases below."""
    parser.add_argument("--vllm-gpu-memory-utilization", dest="vllm_gpu_memory_utilization",
                        type=float, default=0.92)
    parser.add_argument("--vllm-enforce-eager", dest="vllm_enforce_eager",
                        action="store_true", default=False)
    # orchestration flags (excluded from action table)
    parser.add_argument("--router-ip", dest="router_ip", type=str, default=None)
    parser.add_argument("--router-port", dest="router_port", type=int, default=None)
    parser.add_argument("--vllm-server-concurrency", dest="vllm_server_concurrency",
                        type=int, default=512)
    return parser


@pytest.mark.unit
def test_action_table_caches(args_mod, monkeypatch):
    """Two calls return the same dict object (cached)."""
    monkeypatch.setattr(args_mod, "_VLLM_CLI_ACTION_TABLE_CACHE", None)
    monkeypatch.setattr(args_mod, "add_vllm_arguments", _realistic_add_vllm_arguments)
    t1 = args_mod.get_vllm_cli_action_table()
    t2 = args_mod.get_vllm_cli_action_table()
    assert t1 is t2


@pytest.mark.unit
def test_action_table_excludes_orchestration(args_mod, monkeypatch):
    """Orchestration dests (router_*, vllm_server_concurrency, vllm_weight_sync_packed)
    are NOT in the forwarding table — vime owns them, doesn't pass to subprocess."""
    monkeypatch.setattr(args_mod, "_VLLM_CLI_ACTION_TABLE_CACHE", None)
    monkeypatch.setattr(args_mod, "add_vllm_arguments", _realistic_add_vllm_arguments)
    table = args_mod.get_vllm_cli_action_table()
    assert "vllm_gpu_memory_utilization" in table
    assert "vllm_enforce_eager" in table
    assert "router_ip" not in table
    assert "router_port" not in table
    assert "vllm_server_concurrency" not in table


@pytest.mark.unit
def test_action_table_flag_format(args_mod, monkeypatch):
    """Returned table maps dest → (subprocess_flag, action). The subprocess flag
    drops the `vllm-` prefix (so vllm subprocess sees `--gpu-memory-utilization`
    not `--vllm-gpu-memory-utilization`)."""
    monkeypatch.setattr(args_mod, "_VLLM_CLI_ACTION_TABLE_CACHE", None)
    monkeypatch.setattr(args_mod, "add_vllm_arguments", _realistic_add_vllm_arguments)
    table = args_mod.get_vllm_cli_action_table()
    flag, action = table["vllm_gpu_memory_utilization"]
    assert flag == "--gpu-memory-utilization"


# ============================================================================
# `vllm_parse_args` TP default computation via temp_parser. We exercise
# the real `vllm_parse_args` with `add_vllm_arguments` replaced by a minimal
# real-shaped registration (no GPU init).
# ============================================================================


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
        sys, "argv",
        ["train.py", "--rollout-num-gpus-per-engine", "4",
         "--vllm-pipeline-parallel-size", "2"],
    )
    ns = args_mod.vllm_parse_args()
    assert ns.vllm_tensor_parallel_size == 2


@pytest.mark.unit
def test_parse_args_records_user_provided(args_mod, monkeypatch):
    """`_vllm_user_provided` and `_vllm_raw_values` populated for vllm-side dests."""
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
    """`parser.set_defaults(vllm_tensor_parallel_size=...)` should populate the
    attribute on `args` even when no `--vllm-tensor-parallel-size` flag is
    registered (vime SKIPS that dest)."""
    # add_vllm_arguments stub does NOT register `vllm_tensor_parallel_size`
    monkeypatch.setattr(args_mod, "add_vllm_arguments", lambda p: p)
    monkeypatch.setattr(sys, "argv", ["train.py", "--rollout-num-gpus-per-engine", "8"])
    ns = args_mod.vllm_parse_args()
    # set_defaults attached the computed default to the namespace
    assert ns.vllm_tensor_parallel_size == 8
