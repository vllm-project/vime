"""vLLM rollout backend argument definitions.

Mirrors slime/backends/sglang_utils/arguments.py:
- Wholesale-imports ``AsyncEngineArgs.add_cli_args(parser)`` with a wrapper that
  prefixes every flag with ``--vllm-`` and every dest with ``vllm_``.
- Adds a small set of vime-specific orchestration extras (router endpoint,
  server concurrency) that are not part of vllm's native CLI.

vllm is launched as a subprocess (``vllm serve``); we forward each
``args.vllm_*`` value that differs from its vllm-side default via
``get_vllm_cli_action_table()`` (consumed by ``vllm_engine.launch_server_process``).
"""

import argparse
import logging
import sys

from vllm.engine.arg_utils import AsyncEngineArgs

from slime.utils.http_utils import _wrap_ipv6

logger = logging.getLogger(__name__)


def _detect_user_provided_dests(parser, argv: list[str]) -> tuple[set[str], dict[str, str]]:
    """Return (user_provided, raw_values) extracted from ``argv``.

    ``user_provided``: dests the user explicitly named on the command line. Lets
    ``launch_server_process`` disambiguate "user accepted the parsed default"
    from "user passed a value that happens to equal the parsed default"
    (e.g. ``--vllm-gpu-memory-utilization 0.92`` to restore vllm's upstream value).

    ``raw_values``: per-dest mapping to the user's literal CLI string. Used when
    forwarding dataclass-backed flags such as ``--vllm-compilation-config`` —
    vllm's parser converts the JSON into a runtime object whose ``asdict()``
    snapshot contains internal/normalized fields the subprocess parser rejects,
    so we forward the original raw string instead.
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


# Dests already managed at vime / megatron level (orchestrator decides them)
# or non-applicable to subprocess `vllm serve` mode. Same intent as
# sglang_utils/arguments.py:43-61 `skipped_args`.
SKIPPED_DESTS = [
    # model identity: hf_checkpoint owns this
    "model",
    "served_model_name",
    "config",
    # tokenizer: vime uses its own
    "tokenizer",
    "tokenizer_mode",
    "tokenizer_revision",
    # security toggle: always-on for vime's curated checkpoints
    "trust_remote_code",
    # seed: vime computes args.seed + rank
    "seed",
    # dtype: vime --fp16 / training config owns this
    "dtype",
    # distributed topology: rollout_num_gpus_per_engine + actor layout own this
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    # network: engine launcher decides per-engine port/host
    "port",
    "host",
    # vime decides this based on training algo, not user CLI
    "enable_return_routed_experts",
]


def add_vllm_router_arguments(parser):
    """vime's vllm-router orchestration flags (not in AsyncEngineArgs)."""
    parser.add_argument(
        "--vllm-router-ip",
        type=str,
        default=None,
        help="IP address of the vLLM router.",
    )
    parser.add_argument(
        "--vllm-router-port",
        type=int,
        default=None,
        help="Port of the vLLM router.",
    )
    parser.add_argument(
        "--vllm-router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout for requests to the vLLM router in seconds.",
    )
    return parser


def _make_add_argument_wrapper(target_add_argument):
    """Return a wrapper around `add_argument` that skips/prefixes flags + dest.

    The wrapper:
      - Drops the call entirely when the canonical dest is in SKIPPED_DESTS.
      - Prefixes every flag (``-x``, ``--foo-bar``) with ``--vllm-``.
      - Prefixes any explicit ``dest=`` with ``vllm_``.
      - Forwards everything else unchanged to ``target_add_argument``.
    """

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

    Wholesale-imports ``AsyncEngineArgs.add_cli_args(parser)`` via a
    monkey-patched ``parser.add_argument`` AND ``parser.add_argument_group``
    wrapper that prefixes every flag with ``--vllm-`` and every dest with
    ``vllm_``, skipping dests listed in ``SKIPPED_DESTS`` (orchestrator-owned
    or non-applicable to subprocess mode).

    Note: vllm's EngineArgs.add_cli_args creates argument groups
    (``parser.add_argument_group(...)``) and adds args to them. We patch both
    ``add_argument`` and ``add_argument_group`` so prefixing happens regardless
    of which path the vllm code takes.
    """
    parser = add_vllm_router_arguments(parser)
    parser.add_argument(
        "--vllm-server-concurrency",
        type=int,
        default=512,
        help="Max concurrent inference requests sent to each vLLM server worker.",
    )
    # vime-only orchestration knob: not part of vllm's CLI but read by
    # UpdateWeightFromDistributed._use_vllm_packed() to choose packed
    # broadcast vs per-bucket NCCL for dense models.
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
        help="Disable packed sync; use per-bucket NCCL via NcclBridge instead.",
    )
    parser.set_defaults(vllm_weight_sync_packed=True)

    old_parser_add_argument = parser.add_argument
    old_parser_add_argument_group = parser.add_argument_group

    def patched_add_argument_group(*g_args, **g_kwargs):
        group = old_parser_add_argument_group(*g_args, **g_kwargs)
        # Patch the group's add_argument so any flag added through it gets prefixed.
        # _ArgumentGroup also has add_argument_group / add_mutually_exclusive_group,
        # but vllm doesn't nest groups in practice; if it ever does, we'd patch them
        # recursively here.
        group.add_argument = _make_add_argument_wrapper(group.add_argument)
        return group

    parser.add_argument = _make_add_argument_wrapper(old_parser_add_argument)
    parser.add_argument_group = patched_add_argument_group
    AsyncEngineArgs.add_cli_args(parser)
    parser.add_argument = old_parser_add_argument
    parser.add_argument_group = old_parser_add_argument_group

    # NOTE: we deliberately do NOT call ``parser.set_defaults(vllm_gpu_memory_utilization=...)``
    # here, because argparse.set_defaults also mutates ``action.default`` — which would
    # then make ``_forward_vllm_cli_args`` think the user accepted the vllm-side default
    # and skip forwarding. vime-preferred defaults (e.g. gpu_memory_utilization=0.55,
    # weight_transfer_config based on colocate) are applied explicitly in
    # ``vllm_engine.launch_server_process``.

    return parser


# vllm-side defaults for legacy migration. Keep in sync with sglang_utils/arguments.py
# (router_request_timeout_secs=14400, server_concurrency=512) and our own
# add_vllm_router_arguments / add_vllm_arguments above.
_VIME_LEGACY_VLLM_DEFAULTS = {
    "vllm_router_request_timeout_secs": 14400,
    "vllm_server_concurrency": 512,
}


def _migrate_legacy_sglang_flags(args) -> None:
    """Migrate ``--sglang-*`` flags to their ``--vllm-*`` equivalents for vllm runs.

    vime previously read ``args.sglang_router_*`` / ``args.sglang_server_concurrency``
    from the vllm code path; renaming those reads broke existing launch scripts. To
    preserve backward compatibility, copy each legacy field onto the corresponding
    ``vllm_*`` attribute whenever (a) the legacy value was actually set by the user,
    and (b) the new ``vllm_*`` value still equals its default.
    """
    legacy_to_vllm = [
        # (legacy_attr, vllm_attr, vllm_default_when_unset)
        ("sglang_router_ip", "vllm_router_ip", None),
        ("sglang_router_port", "vllm_router_port", None),
        (
            "sglang_router_request_timeout_secs",
            "vllm_router_request_timeout_secs",
            _VIME_LEGACY_VLLM_DEFAULTS["vllm_router_request_timeout_secs"],
        ),
        (
            "sglang_server_concurrency",
            "vllm_server_concurrency",
            _VIME_LEGACY_VLLM_DEFAULTS["vllm_server_concurrency"],
        ),
    ]
    user_provided: set[str] = getattr(args, "_vllm_user_provided", set())
    warned_any = False
    for legacy_attr, vllm_attr, vllm_default in legacy_to_vllm:
        legacy_val = getattr(args, legacy_attr, None)
        vllm_val = getattr(args, vllm_attr, vllm_default)
        if legacy_val is None or legacy_val == vllm_default:
            continue  # legacy not set, or already matches default
        # If the user explicitly passed --vllm-* (even to the same value as the
        # default), respect their explicit choice and skip migration.
        if vllm_attr in user_provided:
            continue
        if vllm_val != vllm_default:
            continue  # value differs from default (e.g. YAML override) → keep
        if not warned_any:
            logger.warning(
                "vime: --sglang-* flags are deprecated for the vllm backend. "
                "Migrating to --vllm-* equivalents; please update your launch scripts."
            )
            warned_any = True
        setattr(args, vllm_attr, legacy_val)


def validate_args(args):
    """vllm-specific validation."""
    # Backwards-compat: if a user still passes legacy ``--sglang-router-ip/port``
    # (vime previously reused those names for the vllm-router), migrate the
    # values with a deprecation warning so an external router isn't silently
    # bypassed.
    _migrate_legacy_sglang_flags(args)

    if getattr(args, "vllm_router_ip", None):
        args.vllm_router_ip = _wrap_ipv6(args.vllm_router_ip)
    return


def vllm_parse_args():
    """Parse vllm flags via an independent ArgumentParser + parse_known_args.

    Mirrors ``sglang_parse_args()`` so the merge flow in
    ``slime/utils/arguments.py`` is symmetric.
    Returns an ``argparse.Namespace`` with all attrs prefixed ``vllm_``, plus:
      - ``_vllm_user_provided``: set of dests the user named on argv
      - ``_vllm_raw_values``: per-dest mapping to the user's literal CLI string
        (used by ``launch_server_process`` to forward dataclass-backed flags
        verbatim instead of re-serializing the parsed runtime object).
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_vllm_arguments(parser)
    args, _ = parser.parse_known_args()
    user_provided, raw_values = _detect_user_provided_dests(parser, sys.argv[1:])
    args._vllm_user_provided = user_provided
    args._vllm_raw_values = raw_values
    return args


# Dests that are vime-specific orchestration (not part of `vllm serve` CLI).
# Excluded from get_vllm_cli_action_table() so launch_server_process won't
# try to forward them as command-line flags to the subprocess.
_VIME_ORCHESTRATION_DESTS = frozenset(
    {
        "vllm_router_ip",
        "vllm_router_port",
        "vllm_router_request_timeout_secs",
        "vllm_server_concurrency",
        "vllm_weight_sync_packed",
    }
)


def get_vllm_cli_action_table():
    """Build {vime_dest -> (primary_flag, action)} mapping for forwardable flags.

    Used by ``vllm_engine.launch_server_process`` to forward only the
    ``args.vllm_*`` values that differ from their vllm-side defaults to the
    ``vllm serve`` subprocess as CLI flags.

    Excludes:
      - vime orchestration extras (router endpoint, server concurrency)
      - non-vllm-prefixed actions
    """
    parser = argparse.ArgumentParser(add_help=False)
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
                primary_flag = "--" + s[len("--vllm-"):]
                break
        if primary_flag is None:
            continue
        table[action.dest] = (primary_flag, action)
    return table
