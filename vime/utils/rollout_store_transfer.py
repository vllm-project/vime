from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from vime.utils.remote_batch import (
    MooncakeRemoteBatch,
    get_cached_mooncake_store,
    normalize_store_init_kwargs,
    remove_mooncake_keys,
)

REMOTE_TENSOR_KEYS = ("tokens", "loss_masks")
PARTITIONED_KEYS = (
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "round_number",
    "sample_indices",
    "rollout_ids",
    "rollout_mask_sums",
    "rollout_log_probs",
    "rollout_routed_experts",
    "prompt",
    "teacher_log_probs",
    "metadata",
)
GLOBAL_KEYS = ("raw_reward", "total_lengths")


@dataclass
class RolloutStoreBatch:
    non_tensor_batch: dict[str, np.ndarray] = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)
    remote_batch: MooncakeRemoteBatch | None = None
    tensors: dict[str, torch.Tensor] | None = None

    def materialize_remote_batch(self) -> RolloutStoreBatch:
        if self.remote_batch is not None:
            self.tensors = self.remote_batch.materialize()
            self.remote_batch = None
        return self


def split_rollout_data_by_dp_mooncake_store(
    args: Any,
    data: dict,
    dp_size: int,
    partitions: list,
    micro_batch_indices: list | None = None,
    num_microbatches: int | None = None,
    global_batch_sizes: list | None = None,
    dynamic_global_batch_size: int | None = None,
) -> list[RolloutStoreBatch]:
    if len(partitions) != dp_size:
        raise ValueError(f"expected {dp_size} partitions, got {len(partitions)}")

    store_init_kwargs = normalize_store_init_kwargs(getattr(args, "mooncake_store_init_kwargs", None))
    store = get_cached_mooncake_store(store_init_kwargs)
    transfer_id = uuid.uuid4().hex
    refs: list[RolloutStoreBatch] = []
    try:
        for dp_rank, partition in enumerate(partitions):
            indices = [int(idx) for idx in partition]
            shard = {key: [data[key][idx] for idx in indices] for key in PARTITIONED_KEYS if key in data}
            shard["partition"] = np.asarray(indices, dtype=np.int64)

            meta_info = {key: data[key] for key in GLOBAL_KEYS if key in data}
            if dynamic_global_batch_size is not None:
                meta_info["dynamic_global_batch_size"] = dynamic_global_batch_size
            if global_batch_sizes is not None:
                meta_info["global_batch_sizes"] = global_batch_sizes
            if num_microbatches is not None:
                meta_info["num_microbatches"] = num_microbatches
            if micro_batch_indices is not None:
                meta_info["micro_batch_indices"] = micro_batch_indices[dp_rank]

            remote_tensors, remote_lengths = _extract_remote_tensors(shard)
            meta_info.update(remote_lengths)
            remote_batch = None
            if remote_tensors:
                remote_batch = MooncakeRemoteBatch.from_tensors(
                    remote_tensors,
                    store,
                    prefix=f"vime-rollout/{transfer_id}/dp{dp_rank}",
                    store_init_kwargs=store_init_kwargs,
                )
                meta_info["mooncake_cleanup_keys"] = list(remote_batch.keys_to_cleanup)
                meta_info["mooncake_cleanup_store_kwargs"] = dict(store_init_kwargs)

            try:
                refs.append(
                    RolloutStoreBatch(
                        non_tensor_batch=_dict_to_non_tensors(shard),
                        meta_info=meta_info,
                        remote_batch=remote_batch,
                    )
                )
            except Exception:
                if remote_batch is not None:
                    remote_batch.cleanup()
                raise
    except Exception:
        cleanup_mooncake_store_refs(refs)
        raise
    return refs


def maybe_cleanup_mooncake_store_refs(
    args: Any, refs: list[RolloutStoreBatch] | RolloutStoreBatch, suppress_errors: bool = False
) -> None:
    if getattr(args, "transfer_backend", "ray") != "mooncake_store":
        return
    batches = refs if isinstance(refs, list) else [refs]
    try:
        cleanup_mooncake_store_refs(batches)
    except Exception:
        if not suppress_errors:
            raise


def cleanup_mooncake_store_refs(refs: list[RolloutStoreBatch]) -> None:
    keys: set[str] = set()
    store_init_kwargs = None
    for batch in refs:
        keys.update(batch.meta_info.get("mooncake_cleanup_keys", []))
        if store_init_kwargs is None:
            store_init_kwargs = batch.meta_info.get("mooncake_cleanup_store_kwargs")
    if keys and store_init_kwargs is not None:
        remove_mooncake_keys(get_cached_mooncake_store(store_init_kwargs), sorted(keys))


def rollout_store_batch_to_data(batch: RolloutStoreBatch) -> dict:
    batch.materialize_remote_batch()
    rollout_data = {key: val.tolist() for key, val in batch.non_tensor_batch.items()}
    rollout_data.update(
        {key: val for key, val in batch.meta_info.items() if not key.startswith("mooncake_cleanup_")}
    )
    if batch.tensors:
        for key, tensor in batch.tensors.items():
            lengths = batch.meta_info.get(f"{key}_lengths")
            rollout_data[key] = _tensor_to_row_tensors(tensor, lengths)
    return rollout_data


def _extract_remote_tensors(shard: dict) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    tensors = {}
    lengths = {}
    for key in REMOTE_TENSOR_KEYS:
        if key not in shard:
            continue
        values = shard.pop(key)
        tensor, field_lengths = _list_to_padded_tensor(values, torch.long if key == "tokens" else torch.int)
        tensors[key] = tensor
        lengths[f"{key}_lengths"] = field_lengths
    return tensors, lengths


def _list_to_padded_tensor(values: list, dtype: torch.dtype) -> tuple[torch.Tensor, list[int]]:
    if not values:
        return torch.empty((0, 0), dtype=dtype), []
    tensors = [torch.as_tensor(value, dtype=dtype).reshape(-1) for value in values]
    lengths = [int(tensor.numel()) for tensor in tensors]
    return pad_sequence(tensors, batch_first=True, padding_value=0), lengths


def _tensor_to_row_tensors(tensor: torch.Tensor, lengths: list[int] | None) -> list[torch.Tensor]:
    if tensor.ndim == 2 and lengths is not None:
        return [tensor[idx, : int(length)] for idx, length in enumerate(lengths)]
    return [tensor[idx] for idx in range(tensor.shape[0])]


def _dict_to_non_tensors(data: dict) -> dict[str, np.ndarray]:
    result = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            result[key] = val
        elif isinstance(val, (int, float, bool, np.number)):
            result[key] = np.asarray([val])
        else:
            result[key] = np.asarray(val, dtype=object)
    return result
