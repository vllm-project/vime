from __future__ import annotations

from typing import Any

from vime.utils.types import Sample


async def tau_bench_rm(args, sample: Sample, **kwargs) -> float:
    return sample.reward if sample.reward is not None else 0.0


async def batched_tau_bench_rm(args, samples, **kwargs) -> list[float] | float:
    if isinstance(samples, Sample):
        return samples.reward if samples.reward is not None else 0.0
    return [s.reward if s.reward is not None else 0.0 for s in samples]
