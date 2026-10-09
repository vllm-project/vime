"""Run with torchrun --nproc-per-node=4 on four free NPUs."""

import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

# Prefer this checkout over another editable installation when launched by torchrun.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vime.backends.megatron_utils.update_weight.delta_sync.sparse_gather import (
    GatherWorkspace,
    gather_slot_entries_to_rank0,
)


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    if dist.get_world_size() != 4:
        dist.destroy_process_group()
        raise ValueError("This regression requires exactly four ranks (--nproc-per-node=4)")
    rank = dist.get_rank()
    workspace = GatherWorkspace()
    preserved = []
    cases = [
        [[0, 0, 0], [0, 0, 0], [2, 1, 0], [0, 0, 0]],
        [[1, 0, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 3, 0], [0, 2, 1], [2, 0, 4], [0, 0, 0]],
        [[0, 0, 0]] * 4,
        [[2, 1, 2], [1, 3, 1], [2, 0, 2], [1, 1, 0]],
    ]
    for dtype in (torch.float32, torch.bfloat16):
        for counts in cases:
            pieces = [
                [torch.arange(size, dtype=torch.int32) + r * 100 + slot * 10 for slot, size in enumerate(row)]
                for r, row in enumerate(counts)
            ]
            indices = torch.cat(pieces[rank]).npu()
            values = torch.cat([piece.float().to(dtype) for piece in pieces[rank]]).npu()
            # Cover noncontiguous caller buffers as well as empty participants.
            indices = torch.stack([indices, indices], dim=1)[:, 0]
            values = torch.stack([values, values], dim=1)[:, 0]
            for budget in (None, 12):
                result = gather_slot_entries_to_rank0(
                    indices,
                    values,
                    torch.tensor(counts[rank]),
                    max_round_bytes=budget,
                    workspace=workspace,
                )
                if rank == 0:
                    for slot, (actual_indices, actual_values) in enumerate(result):
                        expected = torch.cat([pieces[r][slot] for r in range(4)])
                        assert torch.equal(actual_indices.cpu(), expected)
                        assert torch.equal(actual_values.cpu(), expected.float().to(dtype))
                        preserved.append((actual_indices, actual_values, expected, dtype))
                else:
                    assert result is None
        dist.barrier()
    if rank == 0:
        for indices, values, expected, dtype in preserved:
            assert torch.equal(indices.cpu(), expected)
            assert torch.equal(values.cpu(), expected.float().to(dtype))
        print("NPU-SPARSE-GATHER PASS: 20 cases; empty/unequal/noncontiguous/split/reused; FP32/BF16", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
