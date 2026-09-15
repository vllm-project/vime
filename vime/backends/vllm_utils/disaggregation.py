"""PD and EPD-specific vLLM deployment sequencing."""

import logging
import uuid
from typing import Any

import ray

from vime.backends.vllm_utils.engine_group import ServerGroup, ServerGroupPlacement
from vime.backends.vllm_utils.vllm_config import ModelConfig

logger = logging.getLogger(__name__)


def start_pd_server_groups(
    model_config: ModelConfig,
    placement: ServerGroupPlacement,
    router_ip: str | None,
    router_port: int | None,
) -> tuple[list[ServerGroup], list[Any]]:
    """Start prefill/decode groups without waiting for final engine initialization."""
    server_groups = []
    init_handles = []
    for group_config in model_config.server_groups:
        group = placement.create(group_config, router_ip, router_port)
        init_handles.extend(placement.start(group))
        server_groups.append(group)
    return server_groups, init_handles


def start_epd_server_groups(
    model_config: ModelConfig,
    placement: ServerGroupPlacement,
    router_ip: str | None,
    router_port: int | None,
) -> tuple[list[ServerGroup], list[Any], list[str]]:
    """Start encoder groups first, then inject their endpoints into LLM groups."""
    server_groups = []
    transfer_overrides = {
        "ec_transfer_config": {
            "ec_connector_extra_config": {
                "shared_storage_path": f"/dev/shm/vime-ec-{uuid.uuid4().hex}",
            },
        },
    }

    encoder_endpoints: list[str] = []
    for group_config in model_config.server_groups:
        if group_config.worker_type != "encoder":
            continue
        group = placement.create(
            group_config,
            router_ip,
            router_port,
            overrides_extra=transfer_overrides,
        )
        handles = placement.start(group)
        if handles:
            ray.get(handles)
        endpoints = ray.get([engine.get_url.remote() for engine in group.engines])
        encoder_endpoints.extend(endpoint for endpoint in endpoints if endpoint is not None)
        server_groups.append(group)

    logger.info("EPD phase 1 done: collected %d encoder endpoints", len(encoder_endpoints))

    init_handles = []
    for group_config in model_config.server_groups:
        if group_config.worker_type == "encoder":
            continue
        overrides_extra = transfer_overrides if group_config.worker_type in ("regular", "prefill") else None
        if overrides_extra is not None and encoder_endpoints:
            overrides_extra = {
                **transfer_overrides,
                "language_only": True,
                "encoder_urls": encoder_endpoints,
            }
        group = placement.create(
            group_config,
            router_ip,
            router_port,
            overrides_extra=overrides_extra,
        )
        init_handles.extend(placement.start(group))
        server_groups.append(group)

    return server_groups, init_handles, encoder_endpoints


def collect_pd_urls(server_groups: list[ServerGroup]) -> tuple[list[tuple[str, None]], list[str]]:
    """Collect static prefill/decode endpoints after engine initialization."""
    prefill_urls = []
    decode_urls = []
    for group in server_groups:
        for engine in group.engines:
            if engine is None:
                continue
            url = ray.get(engine.get_url.remote())
            if not url:
                continue
            if group.worker_type == "prefill":
                prefill_urls.append((url, None))
            elif group.worker_type == "decode":
                decode_urls.append(url)
    return prefill_urls, decode_urls
