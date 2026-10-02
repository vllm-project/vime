"""Opt-in wall-clock spans for the RolloutManager post-rollout boundary.

`RolloutManager.generate()` snapshots its own duration before
`_convert_samples_to_train_data` and `_split_train_data_by_dp` run, and the
trainer only sees the result through `perf/data_preprocess_time`. The region in
between -- sample -> train-data conversion, the DP split, tensorization, the
object-store puts and the rollout metrics aggregation -- is therefore invisible
to every existing timer.

This module measures that region. It is disabled unless ``VIME_ROLLOUT_SPANS=1``
so the default path pays nothing, and it only accumulates ``perf_counter()``
deltas: no tensor work, no extra Ray calls, no change to any value that is
produced or returned.

Spans nest, so the reported ``total_s`` for a parent includes its children.
"""

from __future__ import annotations

import contextlib
import json
import os
import time

_ENABLED = os.environ.get("VIME_ROLLOUT_SPANS") == "1"


class Spans:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextlib.contextmanager
    def span(self, name: str):
        if not _ENABLED:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1


_current = Spans()


def span(name: str):
    return _current.span(name)


def reset() -> Spans:
    _current.totals.clear()
    _current.counts.clear()
    return _current


def emit(rollout_id: int, total_s: float, logger) -> None:
    if not _ENABLED:
        return
    payload = {
        name: {"total_s": round(value, 6), "calls": _current.counts.get(name, 0)}
        for name, value in sorted(_current.totals.items(), key=lambda kv: -kv[1])
    }
    logger.info("ROLLOUT_SPANS rollout=%s total_s=%.6f %s", rollout_id, total_s, json.dumps(payload))
