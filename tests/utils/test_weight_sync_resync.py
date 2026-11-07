"""Multi-rank (gloo) tests for resending a failed weight push.

Four trainer ranks run fake vLLM trainers whose collectives follow vLLM's own
call order, and the rank-0 sender's requests to a fake rollout engine fail at a
chosen point. Every rank must leave ``send_weights_with_resync`` the same way:
all succeed after one full resend, or all raise. A rank left waiting in a
collective shows up as a gloo timeout (or a process that never exits).
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import torch.multiprocessing as mp

NUM_GPUS = 0
WORLD_SIZE = 4
GLOO_TIMEOUT_SECONDS = 10
NUM_TENSORS = 12
IPC_CHUNK = 4


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class FakeEngine:
    """A rollout engine with the session rules of vLLM's ``gpu_worker.py``.

    ``start`` is refused while a session is open, a failed ``update`` ends the
    session, ``finish`` without a session is refused, and the reported version
    changes only through a successful ``finish``. ``fail_at`` ("start",
    "update" or "finish") makes that request fail in the attempts listed in
    ``fail_attempts``; "update" fails on the ``fail_update_index``-th update of
    the attempt.
    """

    def __init__(self, fail_at, fail_attempts, fail_update_index=1):
        self.fail_at = fail_at
        self.fail_attempts = set(fail_attempts)
        self.fail_update_index = fail_update_index
        self.active = False
        self.version = None
        self.attempt = 0
        self.updates_in_attempt = 0
        self.resumed = False
        self.start_weight_update = _Remote(self._start)
        self.start_draft_weight_update = _Remote(self._start)
        self.update_weights = _Remote(self._update)
        self.finish_weight_update = _Remote(self._finish)
        self.get_weight_version = _Remote(lambda: self.version)
        self.continue_generation = _Remote(self._continue)

    def _failing(self, point):
        return self.fail_at == point and self.attempt in self.fail_attempts

    def _start(self):
        if self.active:
            raise RuntimeError("start_weight_update called while a weight update is already active")
        self.attempt += 1
        self.updates_in_attempt = 0
        if self._failing("start"):
            raise RuntimeError("injected start failure")
        self.active = True

    def _update(self, update_info):
        if not self.active:
            raise RuntimeError("start_weight_update must be called before update_weights")
        self.updates_in_attempt += 1
        if self._failing("update") and self.updates_in_attempt == self.fail_update_index:
            self.active = False
            raise RuntimeError("injected update failure")

    def _finish(self, weight_version=None):
        if not self.active:
            raise RuntimeError("finish_weight_update called without a matching start_weight_update")
        if weight_version is not None and self._failing("finish"):
            raise RuntimeError("injected finish failure")  # the session stays open, as in vLLM
        self.active = False
        if weight_version is not None:
            self.version = weight_version

    def _continue(self):
        self.resumed = True


class GatherSource:
    """Each tensor costs one all-reduce on every rank, standing in for the TP gather."""

    def __init__(self):
        self.draft = False

    def metadata(self):
        return [f"w{index}" for index in range(NUM_TENSORS)]

    def __iter__(self):
        import torch
        import torch.distributed as dist

        for index in range(NUM_TENSORS):
            tensor = torch.ones(1)
            dist.all_reduce(tensor)
            yield f"w{index}", tensor


class NcclLikeTrainer:
    """The call order of vLLM's NCCL trainer (``nccl_engine.py``, ``send_weights``).

    Every rank reads the metadata; non-senders only iterate the source (the
    gather) and return without a barrier. The sender starts the session, has the
    engines enter ``update_weights`` (an update rejected before the broadcast
    surfaces here), iterates the source while broadcasting, then finishes.
    """

    def __init__(self, client, source, is_sender):
        self.client = client
        self.source = source
        self.is_sender = is_sender

    def send_weights(self):
        meta = self.source.metadata()
        if not self.is_sender:
            for _ in self.source:
                pass
            return
        self.client.start_weight_update()
        self.client.update_weights({"names": meta})
        for _ in self.source:
            pass
        self.client.finish_weight_update()


class IpcLikeTrainer:
    """The call order of vLLM's IPC trainer (``ipc_engine.py``, ``send_weights``).

    The sender starts the session; every rank iterates the source in chunks and,
    per chunk, all-gathers its IPC handles and passes a barrier before the sender
    ships the chunk; the sender finishes; every rank ends in a barrier.
    """

    def __init__(self, client, source, is_sender):
        self.client = client
        self.source = source
        self.is_sender = is_sender

    def send_weights(self):
        import torch.distributed as dist

        if self.is_sender:
            self.client.start_weight_update()
        chunk = []
        for name, _ in self.source:
            chunk.append(name)
            if len(chunk) == IPC_CHUNK:
                self._ship(chunk)
                chunk = []
        if chunk:
            self._ship(chunk)
        if self.is_sender:
            self.client.finish_weight_update()
        dist.barrier()

    def _ship(self, chunk):
        import torch.distributed as dist

        handles = [None] * dist.get_world_size()
        dist.all_gather_object(handles, list(chunk))
        dist.barrier()
        if self.is_sender:
            self.client.update_weights({"names": list(chunk)})


def _naive_send(trainer):
    """The plan's first sketch: send, then agree on failure. Used as the negative control."""
    import torch
    import torch.distributed as dist

    ok = torch.ones(1, dtype=torch.int32)
    try:
        trainer.send_weights()
    except Exception:
        ok.zero_()
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not ok.item():
        raise RuntimeError("naive: push failed")


def _worker(rank, port, transport, fail_at, fail_attempts, fail_update_index, naive, results):
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE, timeout=timedelta(seconds=GLOO_TIMEOUT_SECONDS))
    from vime.backends.megatron_utils.update_weight import common
    from vime.utils import distributed_utils

    distributed_utils.set_gloo_group(dist.group.WORLD)
    engine = FakeEngine(fail_at, fail_attempts, fail_update_index)
    trainer_cls = NcclLikeTrainer if transport == "nccl" else IpcLikeTrainer
    source = GatherSource()
    reconnects = []

    def make_trainer():
        latch = transport == "ipc" and not naive
        client = common.VimeRayWeightSyncClient([engine], lambda: 1, latch_errors=latch)
        trainer_source = common.ResumableWeightSource(source) if transport == "nccl" else source
        return trainer_cls(client, trainer_source, is_sender=rank == 0)

    def reconnect():
        reconnects.append(1)
        return make_trainer()

    try:
        trainer = make_trainer()
        if naive:
            _naive_send(trainer)
        else:
            common.send_weights_with_resync(trainer, reconnect=reconnect if transport == "nccl" else None)
            common.resume_generation_on_all_ranks([engine], 1)
        outcome = "ok"
    except Exception as error:
        outcome = f"raised: {error}"
    state = (engine.version, engine.resumed, engine.attempt) if rank == 0 else None
    results.put((rank, outcome, len(reconnects), state))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _run(transport, fail_at, fail_attempts, *, fail_update_index=1, naive=False):
    ctx = mp.get_context("spawn")
    results = ctx.SimpleQueue()
    port = _unit_free_port()
    procs = [
        ctx.Process(
            target=_worker,
            args=(rank, port, transport, fail_at, tuple(fail_attempts), fail_update_index, naive, results),
        )
        for rank in range(WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=3 * GLOO_TIMEOUT_SECONDS + 30)
    hung = [rank for rank, proc in enumerate(procs) if proc.is_alive()]
    for proc in procs:
        if proc.is_alive():
            proc.kill()
            proc.join()
    outcomes = {}
    while not results.empty():
        rank, outcome, reconnects, state = results.get()
        outcomes[rank] = (outcome, reconnects, state)
    for rank in hung:
        outcomes.setdefault(rank, ("hung", None, None))
    return outcomes


def _unit_free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.unit
@pytest.mark.parametrize(
    "transport,fail_at,fail_update_index",
    [
        ("ipc", "start", 1),
        ("ipc", "update", 2),
        ("ipc", "finish", 1),
        ("nccl", "start", 1),
        ("nccl", "update", 1),
        ("nccl", "finish", 1),
    ],
)
def test_a_failed_push_is_resent_on_every_rank(transport, fail_at, fail_update_index):
    outcomes = _run(transport, fail_at, [1], fail_update_index=fail_update_index)

    assert sorted(outcomes) == list(range(WORLD_SIZE))
    assert all(outcome == "ok" for outcome, _, _ in outcomes.values()), outcomes
    version, resumed, attempts = outcomes[0][2]
    assert (version, resumed, attempts) == ("1", True, 2)
    # NCCL rebuilds the channel on every rank before the resend; IPC reuses it
    assert {reconnects for _, reconnects, _ in outcomes.values()} == {1 if transport == "nccl" else 0}


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["ipc", "nccl"])
def test_every_rank_raises_when_every_attempt_fails(transport):
    outcomes = _run(transport, "update", [1, 2])

    assert sorted(outcomes) == list(range(WORLD_SIZE))
    assert all("failed on all 2 attempts" in outcome for outcome, _, _ in outcomes.values()), outcomes
    version, resumed, attempts = outcomes[0][2]
    assert (version, resumed, attempts) == (None, False, 2)


@pytest.mark.unit
def test_naive_agreement_after_send_leaves_ranks_waiting():
    """Negative control: with errors raised on the sender (no latching), agreeing after
    ``send_weights`` leaves the other ranks inside the trainer's per-chunk gather."""
    outcomes = _run("ipc", "update", [1], fail_update_index=2, naive=True)

    stuck = [
        rank
        for rank, (outcome, _, _) in outcomes.items()
        if outcome == "hung" or (outcome.startswith("raised") and "naive" not in outcome)
    ]
    assert stuck, outcomes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
