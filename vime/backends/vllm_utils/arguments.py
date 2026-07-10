import argparse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vime.utils.http_utils import _wrap_ipv6


def add_vllm_router_arguments(parser):
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
    parser.add_argument(
        "--router-request-timeout-secs",
        type=int,
        default=14400,
        help="Timeout (seconds) for HTTP requests vime makes to the vllm router.",
    )
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


def add_vllm_arguments(parser):
    parser = add_vllm_router_arguments(parser)
    parser.add_argument("--vllm-server-concurrency", type=int, default=512)
    parser.add_argument(
        "--vllm-enable-deterministic-inference",
        action="store_true",
        default=False,
        help=(
            "Make rollout sampling deterministic. Forwards a per-sample ``seed`` "
            "AND exports ``VLLM_BATCH_INVARIANT=1`` to the vLLM subprocess."
        ),
    )
    _vllm_packed = parser.add_mutually_exclusive_group()
    _vllm_packed.add_argument(
        "--vllm-weight-sync-packed",
        dest="vllm_weight_sync_packed",
        action="store_true",
    )
    _vllm_packed.add_argument(
        "--no-vllm-weight-sync-packed",
        dest="vllm_weight_sync_packed",
        action="store_false",
    )
    parser.set_defaults(vllm_weight_sync_packed=True)

    # Monkey-patch parser to prefix all engine flags with --vllm- / vllm_
    old_add_argument = parser.add_argument
    old_add_argument_group = parser.add_argument_group

    skipped_args = [
        "model",
        "config",
        "trust_remote_code",
        "seed",
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

    def _wrap_add_argument(target_add_argument):
        def wrapper(*name_or_flags, **kwargs):
            canonical = kwargs.get("dest")
            if canonical is None:
                for s in name_or_flags:
                    if isinstance(s, str) and s.startswith("--"):
                        canonical = s[2:].replace("-", "_")
                        break
            if canonical in skipped_args:
                return None

            new_flags = []
            for s in name_or_flags:
                if isinstance(s, str) and s.startswith("-"):
                    new_flags.append(f"--vllm-{s.lstrip('-')}")
                else:
                    new_flags.append(s)

            new_kwargs = kwargs.copy()
            if "dest" in new_kwargs and isinstance(new_kwargs["dest"], str):
                if not new_kwargs["dest"].startswith("vllm_"):
                    new_kwargs["dest"] = f"vllm_{new_kwargs['dest']}"

            return target_add_argument(*new_flags, **new_kwargs)

        return wrapper

    def patched_add_argument_group(*g_args, **g_kwargs):
        group = old_add_argument_group(*g_args, **g_kwargs)
        group.add_argument = _wrap_add_argument(group.add_argument)
        return group

    parser.add_argument = _wrap_add_argument(old_add_argument)
    parser.add_argument_group = patched_add_argument_group
    AsyncEngineArgs.add_cli_args(parser)
    from vllm.entrypoints.openai.cli_args import FrontendArgs

    FrontendArgs.add_cli_args(parser)
    parser.add_argument = old_add_argument
    parser.add_argument_group = old_add_argument_group

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
            "Mutually exclusive with --prefill-num-servers and --rollout-external."
        ),
    )

    return parser


def validate_args(args):
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


def vllm_parse_args():
    """Parse vLLM flags via an independent ArgumentParser + parse_known_args.

    Returns a Namespace with all attrs prefixed ``vllm_``.
    """
    parser = FlexibleArgumentParser(add_help=False)
    add_vllm_arguments(parser)

    # Compute default vllm_tensor_parallel_size from CLI args.
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    temp_parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=1)
    temp_parser.add_argument("--vllm-data-parallel-size", type=int, default=1)
    temp_args, _ = temp_parser.parse_known_args()
    pp_size = temp_args.vllm_pipeline_parallel_size
    dp_size = temp_args.vllm_data_parallel_size
    vllm_tp_size = temp_args.rollout_num_gpus_per_engine // (pp_size * dp_size)
    parser.set_defaults(vllm_tensor_parallel_size=vllm_tp_size)

    args, _ = parser.parse_known_args()
    return args
