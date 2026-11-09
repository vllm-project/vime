"""VIME trajectory and consumption contract for an optional queue adapter."""

import sys
from dataclasses import dataclass, fields
from enum import Enum


def _clock(value: float, name: str) -> None:
    if type(value) not in (int, float) or not 0 <= value <= sys.float_info.max:
        raise ValueError(f"{name} must be finite and non-negative")


def _integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class TrajectoryChild:
    sample_id: int
    rollout_id: int
    weight_versions: tuple[str, ...]
    token_count: int
    response_length: int
    loss_mask_length: int
    reward: float
    rollout_log_probs_length: int | None = None
    top_p_token_offsets: tuple[int, ...] | None = None
    top_p_token_count: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("sample_id", self.sample_id),
            ("rollout_id", self.rollout_id),
            ("token_count", self.token_count),
            ("response_length", self.response_length),
            ("loss_mask_length", self.loss_mask_length),
        ):
            _integer(value, name)
        if not isinstance(self.weight_versions, tuple) or any(
            not isinstance(v, str) or not v for v in self.weight_versions
        ):
            raise ValueError("weight versions must be a tuple of nonempty strings")
        if type(self.reward) not in (int, float):
            raise ValueError("reward must be numeric")
        if self.token_count <= 0 or not 0 <= self.response_length <= self.token_count:
            raise ValueError("invalid trajectory token or response length")
        if self.loss_mask_length != self.response_length:
            raise ValueError("loss mask must cover the response")
        if not -sys.float_info.max <= self.reward <= sys.float_info.max:
            raise ValueError("reward must be finite")
        if self.rollout_log_probs_length is not None and self.rollout_log_probs_length != self.response_length:
            raise ValueError("rollout log probabilities must cover the response")
        if self.rollout_log_probs_length is not None:
            _integer(self.rollout_log_probs_length, "rollout_log_probs_length")
        if (self.top_p_token_offsets is None) != (self.top_p_token_count is None):
            raise ValueError("top-p token ids and offsets must appear together")
        if self.top_p_token_offsets is not None:
            offsets = self.top_p_token_offsets
            _integer(self.top_p_token_count, "top_p_token_count")
            for offset in offsets:
                _integer(offset, "top_p_token_offset")
            if len(offsets) != self.response_length + 1 or offsets[0] != 0 or offsets[-1] != self.top_p_token_count:
                raise ValueError("invalid top-p token offsets")
            if any(left > right for left, right in zip(offsets, offsets[1:], strict=False)):
                raise ValueError("top-p token offsets must be monotonic")


@dataclass(frozen=True)
class TrajectoryGroup:
    schema_version: int
    job_id: str
    restart_epoch: int
    group_id: str
    attempt_id: str
    expected_children: int
    completed_children: int
    sampling_config_digest: str
    rollout_policy_version: str
    children: tuple[TrajectoryChild, ...]
    payload_ref: str
    byte_count: int
    ready: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("schema_version", self.schema_version),
            ("restart_epoch", self.restart_epoch),
            ("expected_children", self.expected_children),
            ("completed_children", self.completed_children),
            ("byte_count", self.byte_count),
        ):
            _integer(value, name)
        if type(self.ready) is not bool:
            raise ValueError("ready must be boolean")
        if not isinstance(self.payload_ref, str):
            raise ValueError("payload_ref must be a string")
        if self.schema_version != 1 or any(
            not isinstance(value, str) or not value
            for value in (
                self.job_id,
                self.group_id,
                self.attempt_id,
                self.sampling_config_digest,
                self.rollout_policy_version,
            )
        ):
            raise ValueError("invalid trajectory group schema or identity")
        if self.restart_epoch < 0 or self.expected_children <= 0:
            raise ValueError("invalid restart epoch or group size")
        if self.completed_children != self.expected_children or len(self.children) != self.expected_children:
            raise ValueError("trajectory group is incomplete")
        if len({child.sample_id for child in self.children}) != len(self.children):
            raise ValueError("duplicate sample_id in trajectory group")
        if not self.rollout_policy_version or any(
            not child.weight_versions or set(child.weight_versions) != {self.rollout_policy_version}
            for child in self.children
        ):
            raise ValueError("missing or mixed rollout policy version")
        if self.byte_count < 0 or (self.ready and (not self.payload_ref or self.byte_count == 0)):
            raise ValueError("ready trajectory group needs payload and byte count")

    def require_consumer_version(self, version: str) -> None:
        if not version or version != self.rollout_policy_version:
            raise ValueError(f"rollout policy {self.rollout_policy_version} does not match consumer {version}")


def group_from_dict(value: dict) -> TrajectoryGroup:
    """Decode JSON lists into the validated group contract."""
    if set(value) != {field.name for field in fields(TrajectoryGroup)}:
        raise ValueError("missing or unknown trajectory group fields")
    children = []
    for child in value["children"]:
        if set(child) != {field.name for field in fields(TrajectoryChild)}:
            raise ValueError("missing or unknown trajectory child fields")
        if not isinstance(child["weight_versions"], (tuple, list)):
            raise ValueError("weight versions must be an array")
        offsets = child["top_p_token_offsets"]
        if offsets is not None and not isinstance(offsets, (tuple, list)):
            raise ValueError("top-p offsets must be an array")
        children.append(
            TrajectoryChild(
                sample_id=child["sample_id"],
                rollout_id=child["rollout_id"],
                weight_versions=tuple(child["weight_versions"]),
                token_count=child["token_count"],
                response_length=child["response_length"],
                loss_mask_length=child["loss_mask_length"],
                reward=child["reward"],
                rollout_log_probs_length=child["rollout_log_probs_length"],
                top_p_token_offsets=None if offsets is None else tuple(offsets),
                top_p_token_count=child["top_p_token_count"],
            )
        )
    return TrajectoryGroup(
        schema_version=value["schema_version"],
        job_id=value["job_id"],
        restart_epoch=value["restart_epoch"],
        group_id=value["group_id"],
        attempt_id=value["attempt_id"],
        expected_children=value["expected_children"],
        completed_children=value["completed_children"],
        sampling_config_digest=value["sampling_config_digest"],
        rollout_policy_version=value["rollout_policy_version"],
        children=tuple(children),
        payload_ref=value["payload_ref"],
        byte_count=value["byte_count"],
        ready=value["ready"],
    )


class Phase(str, Enum):
    READY = "ready"
    LEASED = "leased"
    PREPARED = "prepared"
    TRAINING = "training"
    TRAINED = "trained"
    COMMITTED = "committed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Lease:
    owner: str
    token: str
    generation: int
    expires_at: float


class GroupLedger:
    """Local coordinator state; fetching queue metadata is never a training ack."""

    def __init__(self, group: TrajectoryGroup) -> None:
        if not group.ready:
            raise ValueError("cannot lease a group before its payload is ready")
        self.group = group
        self.phase = Phase.READY
        self.generation = 0
        self.lease: Lease | None = None
        self.consumer_batch_id: str | None = None
        self.commit_id: str | None = None
        self.last_time = 0.0

    def _time(self, now: float) -> None:
        _clock(now, "now")
        if now < self.last_time:
            raise ValueError("coordinator clock moved backwards")
        self.last_time = now

    def acquire(self, owner: str, token: str, now: float, ttl: float) -> Lease:
        self.expire(now)
        _clock(ttl, "ttl")
        if self.phase != Phase.READY or not owner or not token or ttl <= 0:
            raise ValueError("group is unavailable or lease arguments are invalid")
        self.generation += 1
        _clock(now + ttl, "lease deadline")
        self.lease = Lease(owner, token, self.generation, now + ttl)
        self.phase = Phase.LEASED
        return self.lease

    def _check(self, lease: Lease, now: float) -> None:
        self._time(now)
        self.expire(now)
        if self.lease != lease or now >= lease.expires_at:
            raise ValueError("lease is stale or expired")

    def prepare(self, lease: Lease, consumer_batch_id: str, now: float) -> None:
        self._check(lease, now)
        if self.phase != Phase.LEASED or not consumer_batch_id:
            raise ValueError("group cannot be prepared")
        self.consumer_batch_id = consumer_batch_id
        self.phase = Phase.PREPARED

    def start_training(self, lease: Lease, now: float) -> None:
        self._check(lease, now)
        if self.phase != Phase.PREPARED:
            raise ValueError("group must be prepared before training")
        self.phase = Phase.TRAINING

    def commit(self, lease: Lease, commit_id: str, now: float) -> None:
        self._check(lease, now)
        if self.phase != Phase.TRAINED or not commit_id:
            raise ValueError("group has not finished a valid training plan")
        self.commit_id = commit_id
        self.phase = Phase.COMMITTED

    def finish_training(self, lease: Lease, now: float) -> None:
        self._check(lease, now)
        if self.phase != Phase.TRAINING:
            raise ValueError("group is not training")
        self.phase = Phase.TRAINED

    def release_before_training(self, lease: Lease, now: float) -> None:
        self._check(lease, now)
        if self.phase not in (Phase.LEASED, Phase.PREPARED):
            raise ValueError("cannot requeue after training began")
        self.lease = None
        self.consumer_batch_id = None
        self.phase = Phase.READY

    def renew(self, lease: Lease, now: float, ttl: float) -> Lease:
        self._check(lease, now)
        _clock(ttl, "ttl")
        if ttl <= 0 or self.phase not in (Phase.LEASED, Phase.PREPARED, Phase.TRAINING, Phase.TRAINED):
            raise ValueError("cannot renew this lease")
        _clock(now + ttl, "lease deadline")
        self.lease = Lease(lease.owner, lease.token, lease.generation, now + ttl)
        return self.lease

    def mark_unknown(self) -> None:
        if self.phase in (Phase.TRAINING, Phase.TRAINED):
            self.phase = Phase.UNKNOWN

    def expire(self, now: float) -> None:
        self._time(now)
        if self.lease is None or now < self.lease.expires_at:
            return
        if self.phase in (Phase.TRAINING, Phase.TRAINED):
            self.phase = Phase.UNKNOWN
        elif self.phase in (Phase.LEASED, Phase.PREPARED):
            self.phase = Phase.READY
            self.lease = None
            self.consumer_batch_id = None
