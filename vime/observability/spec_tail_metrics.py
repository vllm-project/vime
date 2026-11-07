"""Long-tail rollout metrics for speculative decoding.

Speculative decoding pays off most at the end of a rollout, when only a few
long requests are left and decoding is memory bound. The per-rollout means in
``rollout_metrics._compute_spec_metrics`` hide that, so these metrics split
acceptance by response length and measure how much wall time the tail takes.
"""

from __future__ import annotations

import math

import numpy as np

from vime.utils.types import Sample

# (key, lower percentile, upper percentile) of completion length; the first
# tier also includes its lower bound.
LENGTH_TIERS = (
    ("len_p0_p50", 0, 50),
    ("len_p50_p90", 50, 90),
    ("len_p90_p99", 90, 99),
    ("len_p99_p100", 99, 100),
)
TAIL_FRACTION = 0.1


def compute_spec_length_tier_metrics(samples: list[Sample]) -> dict[str, float]:
    """Token-weighted acceptance length per completion-length tier.

    Acceptance length is ``sum(completion tokens) / sum(verify steps)`` over
    the tier, i.e. tokens produced per target-model forward. Samples without
    verify steps (aborted before decoding, or speculation inactive) carry no
    information and are left out rather than counted as zero.
    """
    infos = [s.spec_info for s in samples if s.spec_info.spec_verify_ct > 0]
    if not infos:
        return {}
    lengths = np.array([info.completion_token_num for info in infos], dtype=np.float64)
    verify = np.array([info.spec_verify_ct for info in infos], dtype=np.float64)

    metrics = {"spec_accept_length/token_weighted": float(lengths.sum() / verify.sum())}
    bounds = np.percentile(lengths, [lo for _, lo, _ in LENGTH_TIERS] + [100])
    for i, (key, _, _) in enumerate(LENGTH_TIERS):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (lengths > lo) & (lengths <= hi)
        if i == 0:
            mask |= lengths == lo
        if mask.any():
            metrics[f"spec_accept_length/{key}"] = float(lengths[mask].sum() / verify[mask].sum())
    return metrics


def compute_tail_wall_metrics(samples: list[Sample], rollout_time: float | None = None) -> dict[str, float]:
    """Share of rollout generation wall time spent on the slowest 10% of samples.

    Per-sample latencies overlap, so they cannot be summed into a share.
    Instead, with ``start`` the first ``vllm_generate`` start of any sample,
    ``end`` the last finish, and ``t90`` the time by which 90% of samples had
    finished their last ``vllm_generate``::

        tail/last_10pct_wall_share = (end - t90) / (end - start)

    Timestamps come from the per-sample trace (``trace_utils``), recorded with
    ``time.time()`` by whichever process ran the request; with rollout spread
    over several hosts this assumes their clocks agree. A sample resumed by
    partial rollout still carries the spans of the step it was aborted in, so
    with ``rollout_time`` only spans that started within ``rollout_time``
    before the last finish are counted. Returns nothing when fewer than two
    samples were traced.
    """
    spans = [span for span in map(_generate_span, samples) if span is not None]
    if len(spans) < 2:
        return {}
    end = max(last_end for _, last_end in spans)
    if rollout_time is not None:
        not_before = end - rollout_time
        spans = [span for span in (_generate_span(s, not_before) for s in samples) if span is not None]
        if len(spans) < 2:
            return {}
    starts = [first_start for first_start, _ in spans]
    finishes = [last_end for _, last_end in spans]
    start = min(starts)
    if end <= start:
        return {}
    finishes.sort()
    t90 = finishes[math.ceil((1 - TAIL_FRACTION) * len(finishes)) - 1]
    return {
        "tail/generate_wall_time": end - start,
        "tail/last_10pct_wall_share": (end - t90) / (end - start),
    }


def _generate_span(sample: Sample, not_before: float | None = None) -> tuple[float, float] | None:
    """(first vllm_generate start, last vllm_generate end) of one sample, ignoring
    events before ``not_before``."""
    trace = getattr(sample, "trace", None)
    if not isinstance(trace, dict):
        return None
    first_start = last_end = None
    for event in trace.get("events") or []:
        if event.get("name") != "vllm_generate":
            continue
        ts = event.get("ts")
        if not isinstance(ts, (int, float)) or (not_before is not None and ts < not_before):
            continue
        if event.get("type") == "span_start" and (first_start is None or ts < first_start):
            first_start = ts
        elif event.get("type") == "span_end" and (last_end is None or ts > last_end):
            last_end = ts
    if first_start is None or last_end is None:
        return None
    return first_start, last_end
