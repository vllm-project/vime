"""Cache refresh policy shared by the weight transports."""


def should_flush_cache(flush_cache_interval: int, weight_version: int, initial_weight_version: int = 0) -> bool:
    """Refresh at startup and every N subsequent syncs; nonpositive preserves KV.

    The first publication flushes even when resuming with a nonzero version.
    The subsequent schedule keeps its phase relative to version 1.
    A full refresh aborts requests; other updates pause them in place.
    """
    return weight_version == initial_weight_version + 1 or (
        flush_cache_interval > 0 and (weight_version - 1) % flush_cache_interval == 0
    )


def weight_update_pause_mode(*, flush_cache: bool, resume_generation: bool) -> str:
    """Abort in-flight requests for a flush or deferred resume. Cache clearing is a separate RPC."""
    return "abort" if flush_cache or not resume_generation else "in_place"
