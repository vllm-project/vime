"""Single-consumer lease and capacity owner, independent of data-plane RPCs."""

import math
import time
import uuid
from dataclasses import dataclass, replace
from threading import RLock

from vime.rollout.transfer_queue_adapter import EncodedGroup
from vime.rollout.transfer_queue_contract import GroupLedger, Lease, Phase


@dataclass
class Entry:
    encoded: EncodedGroup
    ledger: GroupLedger
    tokens: int
    readable: bool = False


class QueueCoordinator:
    def __init__(self, max_groups: int, max_tokens: int, max_bytes: int, lease_s: float):
        if any(type(cap) is not int or cap <= 0 for cap in (max_groups, max_tokens, max_bytes)) or (
            isinstance(lease_s, bool) or not math.isfinite(lease_s) or lease_s <= 0
        ):
            raise ValueError("queue capacities and lease must be positive")
        self.limits = (max_groups, max_tokens, max_bytes)
        self.lease_s = lease_s
        self.entries: dict[str, Entry] = {}
        self.used = [0, 0, 0]
        self.batch: tuple[str, ...] = ()
        self.batch_id: str | None = None
        self.leases: dict[str, Lease] = {}
        self.committed_round = -1
        self.failed = False
        self.lock = RLock()

    def reserve(self, encoded: EncodedGroup) -> bool:
        with self.lock:
            if encoded.collection_round != self.committed_round + 1:
                raise ValueError("late or future group collection round")
            key = encoded.manifest.payload_ref
            if self.failed or self.batch:
                raise RuntimeError("queue admission is paused")
            if key in self.entries:
                if self.entries[key].encoded != replace(encoded, rows=()):
                    raise ValueError("conflicting duplicate trajectory group")
                return False
            tokens = sum(child.token_count for child in encoded.manifest.children)
            delta = (1, tokens, encoded.working_bytes)
            if any(use + add > cap for use, add, cap in zip(self.used, delta, self.limits, strict=True)):
                raise ValueError("complete group or round exceeds queue capacity")
            self.entries[key] = Entry(replace(encoded, rows=()), GroupLedger(encoded.manifest), tokens)
            self.used = [use + add for use, add in zip(self.used, delta, strict=True)]
            return True

    def rollback(self, key: str) -> None:
        with self.lock:
            entry = self.entries[key]
            if entry.ledger.phase != Phase.READY:
                raise ValueError("cannot roll back a leased group")
            del self.entries[key]
            self.used = [
                self.used[0] - 1,
                self.used[1] - entry.tokens,
                self.used[2] - entry.encoded.working_bytes,
            ]

    def mark_readable(self, key: str) -> None:
        with self.lock:
            if self.failed or self.batch:
                raise RuntimeError("queue admission is paused")
            self.entries[key].readable = True

    def seal(self, keys: tuple[str, ...], batch_id: str, policy_version: str) -> None:
        with self.lock:
            if self.failed or self.batch or not keys or len(keys) != len(set(keys)):
                raise ValueError("cannot seal this training plan")
            if set(keys) != set(self.entries):
                raise ValueError("training plan must contain every admitted complete group")
            now = time.monotonic()
            for key in keys:
                if not self.entries[key].readable:
                    raise ValueError("group payload has not been read and validated")
                self.entries[key].encoded.manifest.require_consumer_version(policy_version)
                if self.entries[key].ledger.phase != Phase.READY:
                    raise ValueError("group is unavailable")
            self.batch, self.batch_id = keys, batch_id
            for key in keys:
                ledger = self.entries[key].ledger
                lease = ledger.acquire("actor", uuid.uuid4().hex, now, self.lease_s)
                ledger.prepare(lease, batch_id, now)
                self.leases[key] = lease

    def heartbeat(self) -> None:
        with self.lock:
            now = time.monotonic()
            for key in self.batch:
                self.leases[key] = self.entries[key].ledger.renew(self.leases[key], now, self.lease_s)

    def start_training(self) -> None:
        with self.lock:
            if self.failed or not self.batch:
                raise ValueError("no valid training plan")
            now = time.monotonic()
            for key in self.batch:
                self.entries[key].ledger.start_training(self.leases[key], now)

    def finish_training(self, rollout_id: int, batch_id: str) -> None:
        with self.lock:
            if self.failed or not self.batch or batch_id != self.batch_id or rollout_id != self.committed_round + 1:
                raise ValueError("optimizer acknowledgment does not match the sealed plan")
            now = time.monotonic()
            for key in self.batch:
                self.entries[key].ledger.finish_training(self.leases[key], now)
            for key in self.batch:
                self.entries[key].ledger.commit(self.leases[key], batch_id, now)
            self.committed_round = rollout_id

    def fail(self) -> None:
        with self.lock:
            self.failed = True
            for entry in self.entries.values():
                entry.ledger.mark_unknown()

    def reclaimed(self, key: str) -> None:
        with self.lock:
            entry = self.entries[key]
            if entry.ledger.phase != Phase.COMMITTED:
                raise ValueError("only committed payloads can be reclaimed")
            del self.entries[key]
            self.used = [
                self.used[0] - 1,
                self.used[1] - entry.tokens,
                self.used[2] - entry.encoded.working_bytes,
            ]

    def publication_confirmed(self) -> None:
        with self.lock:
            if self.failed or self.entries:
                raise RuntimeError("publication requires committed, reclaimed groups")
            self.batch, self.batch_id, self.leases = (), None, {}

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            if self.failed or self.entries or self.batch:
                raise RuntimeError("only a quiescent queue can be checkpointed")
            return {"schema_version": 1, "committed_round": self.committed_round}

    def restore(self, snapshot: dict[str, int]) -> None:
        with self.lock:
            if (
                self.entries
                or self.batch
                or type(snapshot["schema_version"]) is not int
                or snapshot["schema_version"] != 1
                or (type(snapshot["committed_round"]) is not int or snapshot["committed_round"] < -1)
            ):
                raise ValueError("unsupported queue recovery state")
            self.committed_round = snapshot["committed_round"]
