"""Version and token alignment contract for collect-only draft features."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftSequence:
    sample_id: int
    group_id: int | None
    rollout_id: int | None
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    weight_versions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.token_ids or len(self.token_ids) != len(self.loss_mask):
            raise ValueError("token_ids and loss_mask must have the same nonzero length")
        if any(mask not in (0, 1) for mask in self.loss_mask) or self.loss_mask[-1]:
            raise ValueError("loss_mask must be binary and cannot select the final token")


@dataclass(frozen=True)
class DraftToken:
    sample_id: int
    token_position: int
    packed_position: int
    target_token_id: int


@dataclass(frozen=True)
class DraftTokenMap:
    sequences: tuple[DraftSequence, ...]
    sequence_offsets: tuple[int, ...]
    selected_tokens: tuple[DraftToken, ...]

    def __post_init__(self) -> None:
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
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    layer_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
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
        if self.round_id < 0:
            raise ValueError("round_id must be nonnegative")
        if self.policy_source_version != self.head_source_version:
            raise ValueError("head snapshot and target features must have the same source version")
        if self.model_tag not in ("actor", "old_actor"):
            raise ValueError("draft features require actor or old_actor target")
        if self.capture_point != "lm_head_input":
            raise ValueError("unsupported capture_point")
        if self.layer_ids:
            raise ValueError("lm_head_input capture does not use decoder layer ids")
        if (self.tp, self.pp, self.cp, self.dp) != (1, 1, 1, 1):
            raise ValueError("draft feature schema v1 supports only TP=PP=CP=DP=1")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("unsupported feature dtype")
        if self.byte_count < 0 or (
            self.ready and (not self.payload_ref or not self.head_snapshot_ref or self.byte_count == 0)
        ):
            raise ValueError("ready feature batch requires payload, head snapshot, and byte count")
