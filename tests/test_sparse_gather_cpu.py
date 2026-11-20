"""Real CPU process-group coverage for sparse gather chunk boundaries."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    path = Path(__file__).resolve().parents[1] / "vime/backends/megatron_utils/delta_sync/sparse_gather.py"
    spec = importlib.util.spec_from_file_location("sparse_gather_cpu", path)
    gather = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gather)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for rows in ([[0], [31]], [[1, 13, 0], [3, 0, 2]], [[0, 0], [0, 0]]):
            for dtype in (torch.bfloat16, torch.float32):
                for budget in (12, 128):
                    pieces = [
                        [torch.arange(n, dtype=torch.int32) + r * 100 + s * 40 for s, n in enumerate(row)]
                        for r, row in enumerate(rows)
                    ]
                    indices = torch.cat(pieces[rank])
                    values = indices.to(dtype)

                    class Workspace(gather.GatherWorkspace):
                        def tensor(self, key, size, reference, budget=budget, indices=indices, values=values):
                            if key in {"recv_indices", "recv_values"}:
                                assert size * (indices.element_size() + values.element_size()) <= budget
                            return super().tensor(key, size, reference)

                    workspace = Workspace()
                    previous = None
                    for _ in range(2):
                        result = gather.gather_slot_entries_to_rank0(
                            indices,
                            values,
                            torch.tensor(rows[rank]),
                            max_round_bytes=budget,
                            workspace=workspace,
                        )
                        if rank == 0:
                            for s, (actual_indices, actual_values) in enumerate(result):
                                expected = torch.cat([pieces[r][s] for r in range(2)])
                                assert torch.equal(actual_indices, expected)
                                assert torch.equal(actual_values, expected.to(dtype))
                                if previous is not None:
                                    assert torch.equal(previous[s][0], expected)
                                    assert torch.equal(previous[s][1], expected.to(dtype))
                            previous = result
                        else:
                            assert result is None
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
def test_sparse_gather_cpu_oversized_slots_and_total_receive_budget(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
