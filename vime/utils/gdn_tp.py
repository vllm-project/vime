"""TP layout helpers for GDN's unequal, concatenated Q/K/V projections."""

import torch


def shard_gdn_qkv(tensor, split_sizes, parallel_size, parallel_rank):
    if any(size % parallel_size for size in split_sizes):
        raise ValueError(f"GDN Q/K/V sizes {split_sizes} must divide TP{parallel_size}")
    return torch.cat(
        [part.chunk(parallel_size, dim=0)[parallel_rank] for part in tensor.split(tuple(split_sizes), dim=0)],
        dim=0,
    ).contiguous()


def merge_gdn_qkv(partitions, split_sizes):
    size = len(partitions)
    local_sizes = tuple(n // size for n in split_sizes)
    parts = [p.split(local_sizes, dim=0) for p in partitions]
    return torch.cat([torch.cat([p[i] for p in parts], dim=0) for i in range(3)], dim=0).contiguous()
