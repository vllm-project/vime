"""Pinned SimpleStorage transport for complete text trajectory groups.

Imported only by the opt-in rollout path. Native fetch is a storage operation;
GroupLedger owns the optimizer acknowledgment boundary.
"""

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vime.rollout.transfer_queue_contract import TrajectoryChild, TrajectoryGroup
from vime.utils.types import Sample

if TYPE_CHECKING:
    from tensordict import TensorDict
    from transfer_queue.client import AsyncTransferQueueClient
    from transfer_queue.metadata import BatchMeta


@dataclass(frozen=True)
class EncodedGroup:
    manifest: TrajectoryGroup
    rows: tuple[bytes, ...]
    digest: str
    collection_round: int
    working_bytes: int = field(compare=False)

    def __post_init__(self) -> None:
        if type(self.working_bytes) is not int or self.working_bytes < self.manifest.byte_count:
            raise ValueError("working bytes must include the native payload")


def _tree_bytes(value: object) -> int:
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(_tree_bytes(key) + _tree_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(_tree_bytes(item) for item in value)
    return size


def encode_group(
    samples: list[Sample],
    *,
    job_id: str,
    restart_epoch: int,
    group_id: str,
    attempt_id: str,
    expected_children: int,
    policy_version: str,
    sampling_digest: str,
    reward_key: str | None,
    collection_round: int = 0,
) -> EncodedGroup:
    rows, children = [], []
    prepared_bytes = 0
    if type(collection_round) is not int or collection_round < 0:
        raise ValueError("invalid collection round")
    if (
        not samples
        or len({sample.group_index for sample in samples}) != 1
        or type(samples[0].group_index) is not int
        or samples[0].group_index < 0
    ):
        raise ValueError("a group must contain children from one prompt draw")
    for sample in samples:
        if sample.index is None:
            raise ValueError("queue samples require sample and rollout identities")
        if (
            sample.multimodal_inputs
            or sample.multimodal_train_inputs
            or sample.multimodal_train_input_id is not None
            or sample.rollout_routed_experts is not None
        ):
            raise ValueError("SimpleStorage v1 supports text trajectories without MoE replay")
        if sample.teacher_log_probs is not None or sample.train_metadata or sample.remove_sample:
            raise ValueError("queue v1 does not support teacher, custom training metadata or removed samples")
        if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
            raise ValueError("only terminal trainable children can enter the queue")
        if not isinstance(sample.weight_versions, (tuple, list)):
            raise ValueError("queue samples require explicit policy provenance")
        if any(type(token) is not int or token < 0 for token in sample.tokens):
            raise ValueError("invalid token ids")
        mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
        if any(type(value) is not int or value not in (0, 1) for value in mask):
            raise ValueError("invalid loss mask")
        if isinstance(sample.reward, dict):
            if reward_key is None:
                raise ValueError("dictionary rewards require a reward key")
            reward = sample.reward[reward_key]
        else:
            reward = sample.reward
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise ValueError("queue children require a finite scalar reward")
        if sample.rollout_log_probs is not None and any(
            type(value) not in (int, float) or not math.isfinite(value) for value in sample.rollout_log_probs
        ):
            raise ValueError("invalid rollout log probabilities")
        ids = (
            None
            if sample.rollout_top_p_token_ids is None
            else torch.as_tensor(sample.rollout_top_p_token_ids).tolist()
        )
        offsets = (
            None
            if sample.rollout_top_p_token_offsets is None
            else torch.as_tensor(sample.rollout_top_p_token_offsets).tolist()
        )
        if ids is not None and any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("invalid top-p token ids")
        child = TrajectoryChild(
            sample.index,
            # Match VIME's existing one-sample-per-rollout fallback. Keep the
            # original nullable field in the payload; never substitute a round id.
            sample.index if sample.rollout_id is None else sample.rollout_id,
            tuple(sample.weight_versions),
            len(sample.tokens),
            sample.response_length,
            len(mask),
            reward,
            None if sample.rollout_log_probs is None else len(sample.rollout_log_probs),
            None if offsets is None else tuple(offsets),
            None if ids is None else len(ids),
        )
        payload = sample.to_dict()
        payload.update(
            weight_versions=list(sample.weight_versions),
            rollout_top_p_token_ids=ids,
            rollout_top_p_token_offsets=offsets,
        )
        prepared_bytes += _tree_bytes(payload) + sys.getsizeof(sample) + sys.getsizeof(sample.__dict__)
        rows.append(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        children.append(child)
    width = max((len(row) for row in rows), default=0)
    partition = f"{job_id}/{restart_epoch}/{group_id}/{attempt_id}"
    manifest = TrajectoryGroup(
        1,
        job_id,
        restart_epoch,
        group_id,
        attempt_id,
        expected_children,
        len(children),
        sampling_digest,
        policy_version,
        tuple(children),
        partition,
        len(rows) * (width + 8),
        True,
    )
    digest = hashlib.sha256(b"\0".join(rows)).hexdigest()
    # Account for owned JSON, decoded Sample fields, and five tensor/transport
    # buffers (put, serialization, storage, fetch, decode). This is an explicit
    # queue working-set budget; allocator/Ray metadata and model memory are not RSS-capped.
    working_bytes = 5 * manifest.byte_count + 2 * (_tree_bytes(rows) + prepared_bytes)
    return EncodedGroup(manifest, tuple(rows), digest, collection_round, working_bytes)


def to_tensor_dict(encoded: EncodedGroup) -> "TensorDict":
    from tensordict import TensorDict

    width = max(map(len, encoded.rows))
    payload = torch.zeros((len(encoded.rows), width), dtype=torch.uint8)
    for i, row in enumerate(encoded.rows):
        payload[i, : len(row)] = torch.frombuffer(bytearray(row), dtype=torch.uint8)
    lengths = torch.tensor([[len(row)] for row in encoded.rows], dtype=torch.int64)
    if payload.numel() + lengths.numel() * lengths.element_size() != encoded.manifest.byte_count:
        raise ValueError("declared payload bytes do not match the encoded tensors")
    return TensorDict({"payload": payload, "length": lengths}, batch_size=[len(encoded.rows)])


def decode_group(data: "TensorDict", encoded: EncodedGroup) -> list[Sample]:
    if set(data.keys()) != {"payload", "length"} or data.batch_size[0] != encoded.manifest.expected_children:
        raise ValueError("TransferQueue returned an empty or incomplete group")
    rows = []
    payloads, lengths = data["payload"].unbind(), data["length"].unbind()
    if sum(row.numel() * row.element_size() for row in (*payloads, *lengths)) != encoded.manifest.byte_count:
        raise ValueError("declared payload bytes do not match the fetched tensors")
    for payload, length in zip(payloads, lengths, strict=True):
        if length.numel() != 1 or length.dtype != torch.int64:
            raise ValueError("TransferQueue returned invalid payload lengths")
        size = int(length.item())
        if payload.dtype != torch.uint8 or payload.device.type != "cpu" or size <= 0 or size > payload.numel():
            raise ValueError("TransferQueue returned invalid payload metadata")
        rows.append(payload[:size].numpy().tobytes())
    if tuple(rows) != encoded.rows or hashlib.sha256(b"\0".join(rows)).hexdigest() != encoded.digest:
        raise ValueError("TransferQueue payload does not match the ready manifest")
    return [Sample.from_dict(json.loads(row)) for row in rows]


class SimpleStorageAdapter:
    def __init__(self, client: "AsyncTransferQueueClient", timeout_s: float):
        self.client = client
        self.timeout_s = timeout_s

    async def put(self, encoded: EncodedGroup) -> "BatchMeta":
        import asyncio

        return await asyncio.wait_for(
            self.client.async_put(to_tensor_dict(encoded), partition_id=encoded.manifest.payload_ref), self.timeout_s
        )

    async def get(self, metadata: "BatchMeta", encoded: EncodedGroup) -> list[Sample]:
        import asyncio

        data = await asyncio.wait_for(self.client.async_get_data(metadata), self.timeout_s)
        return decode_group(data, encoded)

    async def delete(self, encoded: EncodedGroup) -> None:
        import asyncio

        await asyncio.wait_for(self.client.async_clear_partition(encoded.manifest.payload_ref), self.timeout_s)
