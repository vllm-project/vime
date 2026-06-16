"""vLLM rollout backend argument definitions.

Wholesale-imports ``AsyncEngineArgs.add_cli_args`` prefixed ``--vllm-``/``vllm_``,
adds vime orchestration extras, and provides ``get_vllm_cli_action_table`` for
subprocess CLI forwarding (vLLM is launched as ``vllm serve``).
"""

import argparse
import sys

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vime.utils.http_utils import _wrap_ipv6


def _detect_user_provided_dests(parser, argv: list[str]) -> tuple[set[str], dict[str, str]]:
    """Return (user_provided, raw_values) extracted from ``argv``.

    ``user_provided``: dests explicitly named on the CLI — used to distinguish "user
    passed a value equaling the default" from "user accepted the parsed default".

    ``raw_values``: literal CLI strings per dest — used to forward dataclass-backed
    flags (e.g. ``--vllm-compilation-config``) verbatim instead of re-serializing
    the parsed runtime object.
    """
    flag_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for flag in action.option_strings:
            flag_to_dest[flag] = action.dest
    user: set[str] = set()
    raw: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if "=" in token and token.startswith("--"):
            head, raw_val = token.split("=", 1)
            dest = flag_to_dest.get(head)
            if dest is not None:
                user.add(dest)
                raw[dest] = raw_val
            i += 1
            continue
        dest = flag_to_dest.get(token)
        if dest is not None:
            user.add(dest)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                raw[dest] = argv[i + 1]
                i += 2
                continue
        i += 1
    return user, raw


# Dests orchestrator-owned or non-applicable to subprocess `vllm serve` mode.
SKIPPED_DESTS = [
    "model",
    "served_model_name",
    "config",
    "tokenizer",
    "tokenizer_mode",
    "tokenizer_revision",
    "trust_remote_code",
    "seed",
    "dtype",
    # TP is orchestrator-owned; PP/DP remain user-controllable and auto-forward.
    "tensor_parallel_size",
    "nnodes",
    "node_rank",
    "master_addr",
    "master_port",
    "data_parallel_backend",
    "distributed_executor_backend",
    "port",
    "host",
    "enable_return_routed_experts",
]


def add_vllm_router_arguments(parser):
    """vime's vllm-router endpoint flags (host/port are orchestrator-owned, excluded from RouterArgs CLI)."""
    parser.add_argument(
        "--vllm-router-ip",
        type=str,
        default=None,
        help="IP address of the vllm router (where vime connects to send rollout requests).",
    )
    parser.add_argument(
        "--vllm-router-port",
        type=int,
        default=None,
        help="Port of the vllm router.",
    )
    # Bare --router-* namespace: this is a real vllm-router knob (RouterArgs field);
    # host/port use --vllm-router-* because RouterArgs excludes them (exclude_host_port=True).
    parser.add_argument(
        "--router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout (seconds) for HTTP requests vime makes to the vllm router.",
    )
    # dest=router_policy (not vllm_router_policy): flows into RouterArgs.from_cli_args and
    # is read by vllm_rollout.generate to decide whether to send x-session-id headers.
    parser.add_argument(
        "--vllm-router-policy",
        type=str,
        default="consistent_hash",
        dest="router_policy",
        choices=["random", "round_robin", "cache_aware", "power_of_two", "consistent_hash"],
        help=(
            "vllm-router load-balancing policy. Defaults to 'consistent_hash' for "
            "session-affinity routing replay via the x-session-id header."
        ),
    )
    return parser


def _make_add_argument_wrapper(target_add_argument):
    """Return a wrapper that skips dests in SKIPPED_DESTS and prefixes flags/dest with ``vllm-``/``vllm_``."""

    def wrapper(*name_or_flags, **kwargs):
        # determine canonical dest for skip check
        canonical = kwargs.get("dest")
        if canonical is None:
            for s in name_or_flags:
                if isinstance(s, str) and s.startswith("--"):
                    canonical = s[2:].replace("-", "_")
                    break
        if canonical in SKIPPED_DESTS:
            return None

        # prefix flags
        new_flags = []
        for s in name_or_flags:
            if isinstance(s, str) and s.startswith("-"):
                new_flags.append(f"--vllm-{s.lstrip('-')}")
            else:
                new_flags.append(s)

        # prefix dest
        new_kwargs = kwargs.copy()
        if "dest" in new_kwargs and isinstance(new_kwargs["dest"], str):
            if not new_kwargs["dest"].startswith("vllm_"):
                new_kwargs["dest"] = f"vllm_{new_kwargs['dest']}"

        return target_add_argument(*new_flags, **new_kwargs)

    return wrapper


def add_vllm_arguments(parser):
    """Register --vllm-* flags into parser.

    Wholesale-imports ``AsyncEngineArgs.add_cli_args`` via a monkey-patched
    ``parser.add_argument`` / ``parser.add_argument_group`` wrapper that prefixes
    every flag with ``--vllm-`` and every dest with ``vllm_``, skipping dests in
    ``SKIPPED_DESTS``. Both are patched because vLLM creates argument groups.
    Pass a ``FlexibleArgumentParser`` so vLLM's ``deprecated`` kwarg is handled
    natively on Python 3.12.
    """
    parser = add_vllm_router_arguments(parser)
    parser.add_argument(
        "--vllm-server-concurrency",
        type=int,
        default=512,
        help="Max concurrent inference requests sent to each vLLM server worker.",
    )
    parser.add_argument(
        "--vllm-enable-deterministic-inference",
        action="store_true",
        default=False,
        help=(
            "Make rollout sampling deterministic. Forwards a per-sample ``seed`` "
            "(derived from ``--rollout-seed`` and the sample's index in the group) "
            "AND exports ``VLLM_BATCH_INVARIANT=1`` to the vLLM subprocess so attention "
            "/ comm / MM kernels pick batch-invariant variants. Both are required for "
            "true determinism — seed alone does not control kernel selection."
        ),
    )
    parser.add_argument(
        "--vllm-tool-call-parser",
        dest="vllm_tool_call_parser",
        type=str,
        default=None,
        help="vLLM tool-call parser name for agent output parsing (e.g. qwen3_coder).",
    )
    _vllm_packed = parser.add_mutually_exclusive_group()
    _vllm_packed.add_argument(
        "--vllm-weight-sync-packed",
        dest="vllm_weight_sync_packed",
        action="store_true",
        help=(
            "Use one-shot packed weight transfer for dense models (no MoE experts). "
            "Automatically disabled for MoE or compressed-tensors quantization."
        ),
    )
    _vllm_packed.add_argument(
        "--no-vllm-weight-sync-packed",
        dest="vllm_weight_sync_packed",
        action="store_false",
        help="Disable packed sync; send weights per bucket via in-process NCCL (non-packed mode).",
    )

    parser.set_defaults(vllm_weight_sync_packed=True)

    old_parser_add_argument = parser.add_argument
    old_parser_add_argument_group = parser.add_argument_group

    def patched_add_argument_group(*g_args, **g_kwargs):
        group = old_parser_add_argument_group(*g_args, **g_kwargs)
        group.add_argument = _make_add_argument_wrapper(group.add_argument)
        return group

    parser.add_argument = _make_add_argument_wrapper(old_parser_add_argument)
    parser.add_argument_group = patched_add_argument_group
    AsyncEngineArgs.add_cli_args(parser)
    parser.add_argument = old_parser_add_argument
    parser.add_argument_group = old_parser_add_argument_group

    # Deliberately no set_defaults for vllm flags: argparse.set_defaults mutates action.default,
    # which would make _forward_vllm_cli_args skip forwarding values that equal the default.
    # vime-preferred defaults are applied explicitly in vllm_engine.launch_server_process.

    # PD disaggregation / multi-group config
    parser.add_argument(
        "--prefill-num-servers",
        type=int,
        default=None,
        help="Number of prefill servers for PD disaggregation.",
    )
    parser.add_argument(
        "--vllm-config",
        type=str,
        default=None,
        dest="vllm_config",
        help=(
            "Path to a YAML config file for fine-grained vLLM rollout engine deployment. "
            "Enables multi-model serving, PD disaggregation, and heterogeneous server groups. "
            "Mutually exclusive with --prefill-num-servers and --rollout-external."
        ),
    )

    return parser


def validate_args(args):
    """vLLM-specific validation."""
    args.vllm_dp_size = args.vllm_data_parallel_size
    args.vllm_pp_size = args.vllm_pipeline_parallel_size

    if getattr(args, "vllm_router_ip", None):
        args.vllm_router_ip = _wrap_ipv6(args.vllm_router_ip)

    assert not (
        getattr(args, "prefill_num_servers", None) is not None and getattr(args, "rollout_external", False)
    ), "prefill_num_servers cannot be set with --rollout-external-engine-addrs."

    assert not (
        getattr(args, "vllm_config", None) is not None and getattr(args, "rollout_external", False)
    ), "vllm_config cannot be set with --rollout-external-engine-addrs."

    assert not (
        getattr(args, "vllm_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
    ), "vllm_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."


def _sanitize_non_primitive_defaults(args, user_provided: set[str]) -> None:
    """Replace non-primitive default values with None.

    vllm's ``AsyncEngineArgs.add_cli_args`` registers parameters whose defaults
    are complex dataclass/config instances (e.g. ``EPLBConfig()``,
    ``CompilationConfig()``).  When ``parse_known_args`` resolves those defaults
    they become live Python objects on the returned Namespace.

    The main ``parse_args`` in ``slime/utils/arguments.py`` later merges
    *every* attribute from this Namespace into the global ``args`` via
    ``setattr``.  Downstream code (megatron-bridge, model_provider, …) may
    iterate ``vars(args)`` and choke on unrecognised types.

    This function walks the parsed Namespace and replaces any non-primitive
    *default* value (i.e. a dest the user did NOT explicitly provide) with
    ``None``.  User-provided values are intentionally left intact — they are
    forwarded to the vllm subprocess via ``_vllm_raw_values`` and do not need
    the parsed object representation in the main args namespace.
    """
    _PRIMITIVE = (str, int, float, bool, type(None))

    for key, value in list(vars(args).items()):
        if key.startswith("_"):
            continue
        if key in user_provided:
            continue
        if isinstance(value, _PRIMITIVE):
            continue
        if isinstance(value, (list, tuple)):
            if all(isinstance(v, _PRIMITIVE) for v in value):
                continue
        setattr(args, key, None)


def vllm_parse_args():
    """Parse vLLM flags via an independent ArgumentParser + parse_known_args.

    Returns a Namespace with all attrs prefixed ``vllm_``, plus:
      - ``_vllm_user_provided``: set of dests the user named on argv
      - ``_vllm_raw_values``: literal CLI strings per dest (for verbatim forwarding
        of dataclass-backed flags like ``--vllm-compilation-config``)
    """
    parser = FlexibleArgumentParser(add_help=False)
    add_vllm_arguments(parser)

    # Compute default vllm_tensor_parallel_size from CLI args
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    temp_parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=1)
    temp_args, _ = temp_parser.parse_known_args()
    pp_size = temp_args.vllm_pipeline_parallel_size
    vllm_tp_size = temp_args.rollout_num_gpus_per_engine // pp_size
    parser.set_defaults(vllm_tensor_parallel_size=vllm_tp_size)

    args, _ = parser.parse_known_args()
    user_provided, raw_values = _detect_user_provided_dests(parser, sys.argv[1:])
    _sanitize_non_primitive_defaults(args, user_provided)
    args._vllm_user_provided = user_provided
    args._vllm_raw_values = raw_values
    return args


# Dests that are vime orchestration flags (not part of `vllm serve` CLI) — excluded
# from get_vllm_cli_action_table() so launch_server_process won't forward them.
_VIME_ORCHESTRATION_DESTS = frozenset(
    {
        "vllm_router_ip",
        "vllm_router_port",
        "router_request_timeout_secs",
        "router_policy",
        "vllm_server_concurrency",
        "vllm_enable_deterministic_inference",
        "vllm_weight_sync_packed",
        "vllm_tool_call_parser",
        "vllm_config",
        "prefill_num_servers",
    }
)


_VLLM_CLI_ACTION_TABLE_CACHE: dict[str, tuple[str, argparse.Action]] | None = None


def get_vllm_cli_action_table():
    """Build {vime_dest -> (primary_flag, action)} mapping for forwardable flags.

    Used by ``vllm_engine.launch_server_process`` to forward ``args.vllm_*`` values
    that differ from vllm-side defaults to the ``vllm serve`` subprocess.

    Excludes vime orchestration dests and non-vllm-prefixed actions. Cached after
    first build — rebuilding the parser is expensive.
    """
    global _VLLM_CLI_ACTION_TABLE_CACHE
    if _VLLM_CLI_ACTION_TABLE_CACHE is not None:
        return _VLLM_CLI_ACTION_TABLE_CACHE

    parser = FlexibleArgumentParser(add_help=False)
    add_vllm_arguments(parser)

    table: dict[str, tuple[str, argparse.Action]] = {}
    for action in parser._actions:
        if action.dest in _VIME_ORCHESTRATION_DESTS:
            continue
        if not action.dest.startswith("vllm_"):
            continue
        # Pick the first ``--vllm-xxx`` flag (skip ``--no-vllm-xxx`` companions).
        primary_flag = None
        for s in action.option_strings:
            if s.startswith("--vllm-") and not s.startswith("--no-vllm-"):
                primary_flag = "--" + s[len("--vllm-") :]
                break
        if primary_flag is None:
            continue
        table[action.dest] = (primary_flag, action)
    _VLLM_CLI_ACTION_TABLE_CACHE = table
    return table
