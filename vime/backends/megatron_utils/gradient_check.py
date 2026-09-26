"""Opt-in dense Megatron gradient snapshots and offline parallel replay checks.

Snapshots contain only the reduced DP slice, before optimizer clipping/unscaling.
No diagnostic collectives or writes to live gradient buffers are performed.
"""

import argparse
import math
from pathlib import Path

import torch


def save_gradient_snapshot(args, model, directory):
    """Save one rank after forward/backward has finalized gradient reductions.

    Supports dense BF16/FP32 DDP with one distributed optimizer instance. The
    caller must use a fresh directory shared by ranks for each training step.
    """
    import torch.distributed as dist
    from megatron.core import mpu

    from vime.backends.megatron_utils.update_weight.common import named_params_and_buffers

    if getattr(args, "fp16", False) or getattr(args, "fp8", None) or getattr(args, "num_experts", None):
        raise ValueError("Gradient snapshots currently require dense BF16/FP32 models")
    if not getattr(args, "untie_embeddings_and_output_weights", True):
        raise ValueError("Gradient snapshots currently require untied embeddings and output weights")
    tensors = {}
    slices = {}
    for chunk in model:
        config = chunk.ddp_config
        if config.num_distributed_optimizer_instances != 1:
            raise ValueError("Gradient snapshots require one distributed optimizer instance")
        for buffer in chunk.buffers:
            group = buffer.data_parallel_group
            dp_size = dist.get_world_size(group) if config.use_distributed_optimizer else 1
            dp_rank = dist.get_rank(group) if config.use_distributed_optimizer else 0
            for bucket in buffer.buckets:
                size = bucket.grad_data.numel()
                if size % dp_size:
                    raise ValueError("Gradient bucket is not divisible by its DP group")
                lower, upper = dp_rank * (size // dp_size), (dp_rank + 1) * (size // dp_size)
                for param in bucket.params_list:
                    begin, end, _ = buffer.param_index_map[param]
                    begin, end = begin - bucket.offset, end - bucket.offset
                    start, stop = max(begin, lower), min(end, upper)
                    # Empty intersections still carry layout/coverage metadata.
                    start = min(max(start, begin), end)
                    stop = max(start, stop)
                    slices[param] = (start - begin, bucket.grad_data[start:stop].detach().cpu().clone())

    for name, param in named_params_and_buffers(args, model):
        if not param.requires_grad:
            continue
        if name in tensors:
            raise ValueError(f"Duplicate logical parameter: {name}")
        if param not in slices:
            raise ValueError(f"No reduced gradient buffer for {name}")
        start, grad = slices[param]
        sharded = (
            getattr(param, "tensor_model_parallel", False) and getattr(param, "parallel_mode", None) != "duplicated"
        )
        tensors[name] = dict(
            shape=tuple(param.shape),
            start=start,
            grad=grad,
            tp_rank=mpu.get_tensor_model_parallel_rank() if sharded else 0,
            tp_size=mpu.get_tensor_model_parallel_world_size() if sharded else 1,
            partition_dim=getattr(param, "partition_dim", 0) if sharded else 0,
            partition_stride=getattr(param, "partition_stride", 1) if sharded else 1,
            glu=bool(getattr(args, "swiglu", False) and "linear_fc1." in name),
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    torch.save(
        dict(version=1, rank=rank, world_size=dist.get_world_size(), tensors=tensors), directory / f"rank-{rank}.pt"
    )


def _assert_close(actual, expected, name, ranks, *, rtol, atol):
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}; ranks={ranks}")
    actual, expected = actual.double(), expected.double()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    close = finite & torch.isclose(actual, expected, rtol=rtol, atol=atol)
    if not close.all():
        index = tuple((~close).nonzero()[0].tolist())
        error = (actual - expected).abs()
        raise AssertionError(
            f"{name}: index={index}, actual={actual[index].item():.8g}, expected={expected[index].item():.8g}, "
            f"max_abs_error={error.max().item():.8g}, mismatched={(~close).sum().item()}/{close.numel()}, "
            f"rtol={rtol}, atol={atol}; ranks={ranks}"
        )


def _load_snapshot(directory, *, rtol, atol):
    files = list(Path(directory).glob("rank-*.pt"))
    if not files:
        raise ValueError(f"No gradient rank files in {directory}")
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
    world_size = payloads[0]["world_size"]
    if len(payloads) != world_size or {p["rank"] for p in payloads} != set(range(world_size)):
        raise ValueError(f"Missing or duplicate ranks in {directory}")
    records = {}
    for payload in payloads:
        if payload["version"] != 1 or payload["world_size"] != world_size:
            raise ValueError(f"Inconsistent snapshot metadata in {directory}")
        for name, record in payload["tensors"].items():
            records.setdefault(name, []).append((payload["rank"], record))
    if not records:
        raise ValueError(f"Gradient snapshot is empty: {directory}")
    result = {}
    for name, entries in records.items():
        first = entries[0][1]
        shape, tp_size, dim, stride = (first[k] for k in ("shape", "tp_size", "partition_dim", "partition_stride"))
        size = math.prod(shape)
        if not size or tp_size < 1 or stride not in (1, 2):
            raise ValueError(f"{name}: unsupported shape or TP layout")
        if tp_size > 1 and not 0 <= dim < len(shape):
            raise ValueError(f"{name}: invalid partition dimension {dim}")
        shards = [torch.empty(size, dtype=torch.float32) for _ in range(tp_size)]
        covered = [torch.zeros(size, dtype=torch.bool) for _ in range(tp_size)]
        ranks = [rank for rank, _ in entries]
        for rank, record in entries:
            if any(record[k] != first[k] for k in ("shape", "tp_size", "partition_dim", "partition_stride", "glu")):
                raise ValueError(f"{name}: inconsistent shard metadata; rank={rank}")
            tp, start, value = record["tp_rank"], record["start"], record["grad"].float()
            stop = start + value.numel()
            if not 0 <= tp < tp_size or not 0 <= start <= stop <= size or value.ndim != 1:
                raise ValueError(f"{name}: invalid gradient range; rank={rank}")
            if not torch.isfinite(value).all():
                raise AssertionError(f"{name}: nonfinite gradient; rank={rank}")
            overlap = covered[tp][start:stop]
            if overlap.any():
                _assert_close(
                    value[overlap], shards[tp][start:stop][overlap], name + " replica", ranks, rtol=rtol, atol=atol
                )
            shards[tp][start:stop] = value
            covered[tp][start:stop] = True
        if not all(mask.all() for mask in covered):
            raise ValueError(f"{name}: incomplete gradient coverage; ranks={ranks}")
        shards = [shard.reshape(shape) for shard in shards]
        if tp_size == 1:
            tensor = shards[0]
        elif first["glu"]:
            # Megatron GLU shards store [gate_i, up_i], even when stride=1.
            if dim != 0 or shape[0] % 2:
                raise ValueError(f"{name}: invalid GLU layout")
            parts = [shard.chunk(2, dim=0) for shard in shards]
            tensor = torch.cat([part[0] for part in parts] + [part[1] for part in parts], dim=0)
        else:
            if stride != 1:
                raise ValueError(f"{name}: unsupported partition stride {stride}")
            tensor = torch.cat(shards, dim=dim)
        result[name] = (tensor, ranks)
    return result


def compare_snapshots(reference, actual, *, rtol=1e-3, atol=1e-6):
    """Assert complete parameter-wise parity; return the number of tensors checked."""
    if not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    expected = _load_snapshot(reference, rtol=rtol, atol=atol)
    observed = _load_snapshot(actual, rtol=rtol, atol=atol)
    if expected.keys() != observed.keys():
        raise AssertionError(
            f"Parameter names differ: missing={sorted(expected.keys() - observed.keys())}, "
            f"extra={sorted(observed.keys() - expected.keys())}"
        )
    for name in sorted(expected):
        _assert_close(observed[name][0], expected[name][0], name, observed[name][1], rtol=rtol, atol=atol)
    return len(expected)


def compare_runs(reference, actual, *, rtol=1e-3, atol=1e-6):
    """Compare all role/rollout/step snapshots, rejecting missing or extra steps."""
    reference, actual = Path(reference), Path(actual)
    left = {path.parent.relative_to(reference) for path in reference.rglob("rank-*.pt")}
    right = {path.parent.relative_to(actual) for path in actual.rglob("rank-*.pt")}
    if not left or left != right:
        raise ValueError(f"Snapshot steps differ or are empty: reference={sorted(left)}, actual={sorted(right)}")
    return sum(compare_snapshots(reference / step, actual / step, rtol=rtol, atol=atol) for step in sorted(left))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("actual")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    cli_args = parser.parse_args()
    count = compare_runs(cli_args.reference, cli_args.actual, rtol=cli_args.rtol, atol=cli_args.atol)
    print(f"Compared {count} parameter gradients across all saved steps")
