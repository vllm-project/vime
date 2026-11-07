"""Version and token alignment contract for collect-only draft features."""

from dataclasses import dataclass
from typing import cast


def normalize_weight_versions(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(not isinstance(version, str) or not version for version in value):
        raise ValueError("weight_versions must be a list or tuple of nonempty strings")
    return tuple(value)


def _nonnegative(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


@dataclass(frozen=True)
class DraftSequence:
    sample_id: int
    group_id: int | None
    rollout_id: int | None
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    weight_versions: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonnegative(self.sample_id, "sample_id")
        for name, value in (("group_id", self.group_id), ("rollout_id", self.rollout_id)):
            if value is not None:
                _nonnegative(value, name)
        if not self.token_ids or len(self.token_ids) != len(self.loss_mask):
            raise ValueError("token_ids and loss_mask must have the same nonzero length")
        for token in self.token_ids:
            _nonnegative(token, "token_id")
        if any(type(mask) is not int or mask not in (0, 1) for mask in self.loss_mask) or self.loss_mask[-1]:
            raise ValueError("loss_mask must be binary and cannot select the final token")
        if normalize_weight_versions(self.weight_versions) != self.weight_versions:
            raise ValueError("weight_versions must be normalized before constructing a sequence")


@dataclass(frozen=True)
class DraftToken:
    sample_id: int
    token_position: int
    packed_position: int
    target_token_id: int

    def __post_init__(self) -> None:
        for name, value in (
            ("sample_id", self.sample_id),
            ("token_position", self.token_position),
            ("packed_position", self.packed_position),
            ("target_token_id", self.target_token_id),
        ):
            _nonnegative(value, name)


@dataclass(frozen=True)
class DraftTokenMap:
    sequences: tuple[DraftSequence, ...]
    sequence_offsets: tuple[int, ...]
    selected_tokens: tuple[DraftToken, ...]

    def __post_init__(self) -> None:
        for offset in self.sequence_offsets:
            _nonnegative(offset, "sequence_offset")
        offsets = [0]
        for sequence in self.sequences:
            offsets.append(offsets[-1] + len(sequence.token_ids))
        if tuple(offsets) != self.sequence_offsets or not self.selected_tokens:
            raise ValueError("invalid sequence offsets or empty token map")
        selected = set()
        samples = {sequence.sample_id: (sequence, offsets[i]) for i, sequence in enumerate(self.sequences)}
        if len(samples) != len(self.sequences):
            raise ValueError("sample_id must be unique within a feature batch")
        for token in self.selected_tokens:
            if token.sample_id not in samples:
                raise ValueError("selected token references an unknown sample")
            sequence, start = samples[token.sample_id]
            position = token.token_position
            if position < 0 or position >= len(sequence.token_ids) - 1 or sequence.loss_mask[position] != 1:
                raise ValueError("selected token does not match its source sequence")
            if token.packed_position != start + position or token.target_token_id != sequence.token_ids[position + 1]:
                raise ValueError("selected token does not match its source sequence")
            if (token.sample_id, position) in selected:
                raise ValueError("selected token does not match its source sequence")
            selected.add((token.sample_id, position))


def build_token_map(sequences: tuple[DraftSequence, ...], max_tokens: int) -> DraftTokenMap:
    """Map masked causal targets back to original samples after packing or reorder."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if len({sequence.sample_id for sequence in sequences}) != len(sequences):
        raise ValueError("sample_id must be unique within a feature batch")

    offsets = [0]
    selected = []
    for sequence in sequences:
        start = offsets[-1]
        for position, mask in enumerate(sequence.loss_mask[:-1]):
            if mask:
                selected.append(
                    DraftToken(sequence.sample_id, position, start + position, sequence.token_ids[position + 1])
                )
                if len(selected) > max_tokens:
                    raise ValueError("selected tokens exceed max_tokens")
        offsets.append(start + len(sequence.token_ids))
    if not selected:
        raise ValueError("feature batch has no selected target tokens")
    return DraftTokenMap(sequences, tuple(offsets), tuple(selected))


@dataclass(frozen=True)
class DraftFeatureManifest:
    schema_version: int
    feature_batch_id: str
    run_id: str
    round_id: int
    policy_source_version: str
    head_source_version: str
    model_tag: str
    model_config_digest: str
    tokenizer_digest: str
    capture_point: str
    dtype: str
    token_map: DraftTokenMap
    head_snapshot_ref: str
    payload_ref: str
    byte_count: int
    ready: bool
    hidden_size: int
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    layer_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"unsupported draft feature schema_version: {self.schema_version}")
        if not all(
            (
                self.feature_batch_id,
                self.run_id,
                self.policy_source_version,
                self.model_config_digest,
                self.tokenizer_digest,
            )
        ):
            raise ValueError("feature batch identity, source version, and digests are required")
        _nonnegative(self.round_id, "round_id")
        for name, value in (
            ("feature_batch_id", self.feature_batch_id),
            ("run_id", self.run_id),
            ("policy_source_version", self.policy_source_version),
            ("head_source_version", self.head_source_version),
            ("model_config_digest", self.model_config_digest),
            ("tokenizer_digest", self.tokenizer_digest),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if self.policy_source_version != self.head_source_version:
            raise ValueError("head snapshot and target features must have the same source version")
        if self.model_tag not in ("actor", "old_actor"):
            raise ValueError("draft features require actor or old_actor target")
        if self.capture_point != "lm_head_input":
            raise ValueError("unsupported capture_point")
        if self.layer_ids:
            raise ValueError("lm_head_input capture does not use decoder layer ids")
        if any(type(size) is not int or size != 1 for size in (self.tp, self.pp, self.cp, self.dp)):
            raise ValueError("draft feature schema v1 supports only TP=PP=CP=DP=1")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("unsupported feature dtype")
        _nonnegative(self.byte_count, "byte_count")
        _nonnegative(self.hidden_size, "hidden_size")
        if type(self.ready) is not bool or self.hidden_size == 0:
            raise ValueError("ready must be boolean and hidden_size must be positive")
        if self.ready and (not self.payload_ref or not self.head_snapshot_ref or self.byte_count == 0):
            raise ValueError("ready feature batch requires payload, head snapshot, and byte count")
        element_size = 4 if self.dtype == "float32" else 2
        if self.ready and self.byte_count != len(self.token_map.selected_tokens) * self.hidden_size * element_size:
            raise ValueError("byte_count does not match selected features and dtype")
        for reference in (self.payload_ref, self.head_snapshot_ref):
            if not isinstance(reference, str) or "/" in reference or "\\" in reference or reference in (".", ".."):
                raise ValueError("artifact references must be local filenames")


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_id(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _integers(value: object, field: str) -> tuple[int, ...]:
    return tuple(_integer(item, field) for item in _array(value, field))


def manifest_from_dict(payload: dict[str, object]) -> DraftFeatureManifest:
    """Decode a complete manifest, including nested dataclasses and JSON arrays."""
    payload = _object(payload, "manifest")
    if set(payload) - DraftFeatureManifest.__annotations__.keys():
        raise ValueError("unknown manifest fields")
    mapping = _object(payload["token_map"], "token_map")
    sequences = []
    for value in _array(mapping["sequences"], "sequences"):
        sequence = _object(value, "sequence")
        sequences.append(
            DraftSequence(
                sample_id=_integer(sequence["sample_id"], "sample_id"),
                group_id=_optional_id(sequence["group_id"], "group_id"),
                rollout_id=_optional_id(sequence["rollout_id"], "rollout_id"),
                token_ids=_integers(sequence["token_ids"], "token_ids"),
                loss_mask=_integers(sequence["loss_mask"], "loss_mask"),
                weight_versions=normalize_weight_versions(sequence.get("weight_versions")),
            )
        )
    tokens = []
    for value in _array(mapping["selected_tokens"], "selected_tokens"):
        token = _object(value, "selected_token")
        tokens.append(
            DraftToken(
                sample_id=_integer(token["sample_id"], "sample_id"),
                token_position=_integer(token["token_position"], "token_position"),
                packed_position=_integer(token["packed_position"], "packed_position"),
                target_token_id=_integer(token["target_token_id"], "target_token_id"),
            )
        )
    token_map = DraftTokenMap(
        tuple(sequences), _integers(mapping["sequence_offsets"], "sequence_offsets"), tuple(tokens)
    )
    manifest = DraftFeatureManifest(
        schema_version=_integer(payload["schema_version"], "schema_version"),
        feature_batch_id=_string(payload["feature_batch_id"], "feature_batch_id"),
        run_id=_string(payload["run_id"], "run_id"),
        round_id=_integer(payload["round_id"], "round_id"),
        policy_source_version=_string(payload["policy_source_version"], "policy_source_version"),
        head_source_version=_string(payload["head_source_version"], "head_source_version"),
        model_tag=_string(payload["model_tag"], "model_tag"),
        model_config_digest=_string(payload["model_config_digest"], "model_config_digest"),
        tokenizer_digest=_string(payload["tokenizer_digest"], "tokenizer_digest"),
        capture_point=_string(payload["capture_point"], "capture_point"),
        dtype=_string(payload["dtype"], "dtype"),
        token_map=token_map,
        head_snapshot_ref=_string(payload["head_snapshot_ref"], "head_snapshot_ref"),
        payload_ref=_string(payload["payload_ref"], "payload_ref"),
        byte_count=_integer(payload["byte_count"], "byte_count"),
        ready=_boolean(payload["ready"], "ready"),
        hidden_size=_integer(payload["hidden_size"], "hidden_size"),
        tp=_integer(payload.get("tp", 1), "tp"),
        pp=_integer(payload.get("pp", 1), "pp"),
        cp=_integer(payload.get("cp", 1), "cp"),
        dp=_integer(payload.get("dp", 1), "dp"),
        layer_ids=_integers(payload.get("layer_ids", []), "layer_ids"),
    )
    if not manifest.ready:
        raise ValueError("manifest is not ready")
    return manifest
