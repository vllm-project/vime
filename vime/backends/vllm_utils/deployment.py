"""Launch and connect vLLM rollout deployments."""

import logging
import multiprocessing
import random
import time
from typing import Any

from vime.backends.vllm_utils.disaggregation import collect_pd_urls, start_epd_server_groups, start_pd_server_groups
from vime.backends.vllm_utils.engine_group import RolloutServer, ServerGroupPlacement
from vime.backends.vllm_utils.external import start_external_rollout_servers
from vime.backends.vllm_utils.vllm_config import resolve_vllm_config
from vime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info

logger = logging.getLogger(__name__)


def _start_router(
    args,
    *,
    has_pd_disaggregation: bool = False,
    force_new: bool = False,
    bind: tuple[str, int] | None = None,
    prefill_urls: list | None = None,
    decode_urls: list | None = None,
) -> tuple[str, int, int | None]:
    """Start the rollout HTTP gateway (vllm-router)."""
    if bind is not None:
        router_ip, router_port = bind
    else:
        if not force_new and args.vllm_router_ip is not None:
            return args.vllm_router_ip, args.vllm_router_port, None
        router_ip = _wrap_ipv6(get_host_info()[1])
        if force_new or args.vllm_router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))
        else:
            router_port = args.vllm_router_port

    from vllm_router.router_args import RouterArgs

    from vime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warning"
    router_args.request_timeout_secs = args.vllm_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.vllm_pd_disaggregation = True

    if prefill_urls is not None:
        router_args.prefill_urls = prefill_urls
        router_args.decode_urls = decode_urls

    # Disable circuit breaker to prevent RDMA transfer timeouts from
    # marking workers as dead. Timeouts are transient (PCIe contention under
    # high load) and do not indicate a dead server.
    router_args.disable_circuit_breaker = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(target=run_router, args=(router_args,))
    process.daemon = True
    process.start()
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port, router_args.prometheus_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start configured rollout servers without waiting for final engine initialization."""
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)

    config = resolve_vllm_config(args)
    placement = ServerGroupPlacement(
        args=args,
        pg=pg,
        rollout_pg_offset=_compute_rollout_offset(args),
        megatron_num_gpus=_compute_megatron_num_gpus(args),
    )

    servers: dict[str, RolloutServer] = {}
    encoder_metadata: dict[str, tuple[str, list[str]]] = {}
    pending_init_handles: list[Any] = []

    for model_idx, model_config in enumerate(config.models):
        model_config.resolve(args)
        has_pd = model_config.has_pd_disaggregation

        if has_pd:
            router_ip = _wrap_ipv6(get_host_info()[1])
            router_port = find_available_port(random.randint(3000, 4000))
            prometheus_port = None
            engine_router_ip = engine_router_port = None
        else:
            router_ip, router_port, prometheus_port = _start_router(
                args,
                force_new=(model_idx > 0),
            )
            engine_router_ip, engine_router_port = router_ip, router_port

        if model_idx == 0:
            args.vllm_router_ip = router_ip
            args.vllm_router_port = router_port

        if model_config.has_encoder_disaggregation:
            server_groups, init_handles, encoder_endpoints = start_epd_server_groups(
                model_config,
                placement,
                engine_router_ip,
                engine_router_port,
            )
            encoder_metadata[model_config.name] = (
                server_groups[0].model_path,
                encoder_endpoints,
            )
        else:
            server_groups, init_handles = start_pd_server_groups(
                model_config,
                placement,
                engine_router_ip,
                engine_router_port,
            )

        pending_init_handles.extend(init_handles)

        if has_pd:
            prefill_urls, decode_urls = collect_pd_urls(server_groups)
            _, _, prometheus_port = _start_router(
                args,
                has_pd_disaggregation=True,
                bind=(router_ip, router_port),
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
            )

        servers[model_config.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            prometheus_port=prometheus_port,
            model_name=model_config.name,
            update_weights=model_config.update_weights,
        )

    args.vllm_model_routers = {name: (server.router_ip, server.router_port) for name, server in servers.items()}
    args.vllm_model_encoder_endpoints = encoder_metadata
    return servers, pending_init_handles
