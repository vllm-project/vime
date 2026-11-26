"""Validate vLLM weight digests against the API server's managed rank layout."""

import re
from itertools import product

_KEY_PATTERN = re.compile(r"(dp\d+:pp\d+:pcp\d+:tp\d+):(.+)")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_INSTANCE_PATTERN = re.compile(r"[0-9a-f]{32}")


def expected_rank_prefixes(config: dict) -> set[str]:
    # The global DP index survives local rank remapping; older configs expose the rank.
    dp_start = int(config.get("data_parallel_index", config.get("data_parallel_rank", 0)))
    dp_size = int(config.get("data_parallel_size", 1))
    # External/hybrid load balancers expose only the DP ranks owned by this endpoint.
    local_only = config.get("data_parallel_external_lb") or config.get("data_parallel_hybrid_lb")
    if local_only:
        dp_size = int(config.get("data_parallel_size_local") or 1)
    # An explicitly addressed local worker owns a single DP rank.
    if config.get("data_parallel_rank_local") is not None:
        dp_size = 1
    pp_size = int(config.get("pipeline_parallel_size", 1))
    pcp_size = int(config.get("prefill_context_parallel_size", 1))
    tp_size = int(config.get("tensor_parallel_size", 1))
    if dp_start < 0 or min(dp_size, pp_size, pcp_size, tp_size) < 1:
        raise ValueError("Invalid parallel configuration for weight checking")
    return {
        f"dp{dp}:pp{pp}:pcp{pcp}:tp{tp}"
        for dp, pp, pcp, tp in product(
            range(dp_start, dp_start + dp_size), range(pp_size), range(pcp_size), range(tp_size)
        )
    }


def validate_checksums(response: dict, expected_ranks: list[str]) -> dict[str, str]:
    checksums = response.get("checksums") if isinstance(response, dict) else None
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Weight checker checksum response requires a nonempty checksums object")
    ranks = set()
    for key, digest in checksums.items():
        match = _KEY_PATTERN.fullmatch(key) if isinstance(key, str) else None
        if match is None or not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"Invalid weight checksum entry {key!r}")
        ranks.add(match[1])
    if ranks != set(expected_ranks):
        raise ValueError(
            f"Weight checker rank coverage differs: missing={sorted(set(expected_ranks) - ranks)}, "
            f"unexpected={sorted(ranks - set(expected_ranks))}"
        )
    return dict(checksums)


def validate_worker_ids(response: dict, expected_ranks: list[str]) -> dict[str, str]:
    worker_ids = response.get("worker_ids") if isinstance(response, dict) else None
    if not isinstance(worker_ids, dict) or set(worker_ids) != set(expected_ranks) or not worker_ids:
        raise ValueError(
            "Weight checker requires worker_ids covering every managed rank; "
            "install the paired Vime vLLM patch on the serving engines"
        )
    if any(not isinstance(value, str) or _INSTANCE_PATTERN.fullmatch(value) is None for value in worker_ids.values()):
        raise ValueError("Invalid weight checker worker instance ID")
    return dict(worker_ids)
