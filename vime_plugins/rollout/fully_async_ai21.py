"""Fully-async rollout with the AI21 seams re-attached.

vime's stock :func:`vime.rollout.fully_async_rollout.generate_rollout_fully_async`
bypasses three hooks that only exist inside the synchronous
:func:`vime.rollout.vllm_rollout.generate_rollout_async` loop:

  - ``--dynamic-sampling-filter-path`` → the AI21 snoozing curriculum
    (``vime_plugins.filters.snoozing``) and its filtered-out JSONL dump
  - ``--rollout-all-samples-process-path`` → the AI21 prefilter-metrics capture
    (``vime_plugins.metrics.rollout_metrics.capture_prefilter_metrics``)
  - the :class:`MetricGatherer` → ``rollout/dynamic_filter/drop_*`` counts
  - ``--rollout-sample-filter-path`` → kept-batch post-filter

This module keeps the stock :class:`AsyncRolloutWorker` (public API — the
background thread that holds a fixed pool of in-flight generations across
rollout boundaries and requeues ABORTED groups) and re-implements only the
*collector*, applying the same filter/hook semantics as the synchronous path:
each completed group is run through the dynamic filter; dropped groups are
counted and excluded from training but still included in the "all samples"
batch handed to ``--rollout-all-samples-process-path`` (ai21-verl's
"unfiltered rollouts batch"). Groups completed beyond the step's target are
pushed back to the worker queue untouched — they are consumed (and filtered)
by the *next* rollout, so no generated group is dropped or double-filtered.

Wire with::

    --rollout-function-path vime_plugins.rollout.fully_async_ai21.generate_rollout_fully_async_ai21
    --eval-function-path vime.rollout.vllm_rollout.generate_rollout

(``--eval-function-path`` defaults to the rollout function path, and the
fully-async path does not support evaluation — pin eval back to the stock
synchronous implementation.)

Semantics vs the synchronous AI21 regression path (accepted deltas):

  - Off-policy: in-flight generations span weight updates, so a group may mix
    responses from older policies. That is the point of fully-async.
  - ABORTED groups never reach the filter or the prefilter metrics (the worker
    requeues them before the collector sees them), so
    ``response/aborted_ratio/prefilter`` is ~0 here by construction.
  - The single-slot stash in ``rollout_metrics`` stays safe: ``RolloutManager``
    is a Ray actor, so ``generate()`` calls serialize even under
    ``train_async.py``'s overlapped driver — capture and log still run
    back-to-back within one call.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
import time

from vime.rollout.base_types import RolloutFnTrainOutput
from vime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from vime.utils.async_utils import get_async_loop, run
from vime.utils.misc import load_function
from vime.utils.types import Sample

# NOTE: AsyncRolloutWorker is imported lazily inside _make_worker —
# vime.rollout.fully_async_rollout pulls in vllm_rollout (vllm_router, image
# processing deps) which only exists in the training image, and the collector
# logic below is unit-tested on CPU without it.

__all__ = [
    "generate_rollout_fully_async_ai21",
]

logger = logging.getLogger("vime_plugins.rollout.fully_async_ai21")


# Own global worker (same lifecycle as the stock module's, kept separate so the
# plugin only depends on the public AsyncRolloutWorker class).
_global_worker = None
_worker_lock = threading.Lock()


def _make_worker(args, data_buffer, concurrency: int):
    """Build an AsyncRolloutWorker whose loop runs on vime's SHARED background
    event loop (``vime.utils.async_utils``) instead of a private thread+loop.

    The stock worker calls ``asyncio.run`` in its own thread, which binds the
    ``GenerateState`` singleton's semaphore to that private loop; any later
    coroutine on the shared loop — eval via the synchronous
    ``generate_rollout``, most notably — then dies with
    ``RuntimeError: ... is bound to a different event loop``. Scheduling the
    worker loop on the shared loop keeps every consumer of GenerateState on one
    loop, exactly like the synchronous path.
    """
    from vime.rollout.fully_async_rollout import AsyncRolloutWorker

    class SharedLoopAsyncRolloutWorker(AsyncRolloutWorker):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._loop_future = None

        def start(self) -> None:
            if self._loop_future is None or self._loop_future.done():
                self._loop_future = asyncio.run_coroutine_threadsafe(self._loop(), get_async_loop().loop)

        def stop(self) -> None:
            self.running = False
            if self._loop_future is not None:
                try:
                    self._loop_future.result(timeout=5)  # same bound as the stock thread join
                except Exception:  # noqa: BLE001 - still draining, or loop already gone at exit
                    self._loop_future.cancel()
                self._loop_future = None

        def is_alive(self) -> bool:
            return self._loop_future is not None and not self._loop_future.done()

    return SharedLoopAsyncRolloutWorker(args, data_buffer, concurrency=concurrency)


def _get_global_worker(args, data_buffer):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.is_alive():
            logger.info("starting AI21 fully-async rollout worker (on the shared event loop)")
            num_engines = max(1, args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
            _global_worker = _make_worker(args, data_buffer, concurrency=args.vllm_server_concurrency * num_engines)
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


def _group_sort_key(group: list[Sample]) -> int:
    for s in group:
        idx = getattr(s, "index", None)
        if idx is not None:
            return int(idx)
    return 0


async def _collect_filtered(args, rollout_id: int, worker, data_buffer) -> RolloutFnTrainOutput:
    """Drain the worker queue until ``rollout_batch_size`` groups SURVIVE the
    dynamic filter; mirror the synchronous loop's hook semantics throughout."""
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    target = args.rollout_batch_size
    data: list[list[Sample]] = []
    all_data: list[list[Sample]] = []
    leftovers: list[tuple[int, list[Sample]]] = []

    logger.info(
        "AI21 fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(data) < target:
        drained = worker.get_completed_groups()
        if not drained:
            await asyncio.sleep(0.05)
        for gid, group in drained:
            if len(data) >= target:
                # Beyond this step's quota: hand back untouched (unfiltered),
                # so the next rollout filters it against fresh snooze state.
                leftovers.append((gid, group))
                continue
            all_data.append(group)
            filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue
            data.append(group)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "AI21 fully-async rollout %d: kept %d/%d (generated %d), queue=%d, elapsed=%.1fs",
                rollout_id,
                len(data),
                target,
                len(all_data),
                worker.queue_size(),
                now - started,
            )
            last_log = now

    for item in leftovers:
        worker.output_queue.put(item)

    data = sorted(data, key=_group_sort_key)
    all_data = sorted(all_data, key=_group_sort_key)

    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_data, data_buffer)

    logger.info(
        "AI21 fully-async rollout %d: done in %.1fs, generated=%d kept=%d requeued=%d queue_left=%d",
        rollout_id,
        time.time() - started,
        len(all_data),
        len(data),
        len(leftovers),
        worker.queue_size(),
    )
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())


def generate_rollout_fully_async_ai21(args, rollout_id, data_buffer, evaluation: bool = False):
    """vime ``--rollout-function-path`` entrypoint."""
    if evaluation:
        raise ValueError(
            "fully-async rollout doesn't support evaluation mode; "
            "set --eval-function-path vime.rollout.vllm_rollout.generate_rollout"
        )
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)
    return run(_collect_filtered(args, rollout_id, worker, data_buffer))
