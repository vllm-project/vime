import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MODULE_PATH = (
    Path(__file__).parents[5]
    / "vime/backends/megatron_utils/update_weight/sparse_gather.py"
)
SPEC = importlib.util.spec_from_file_location("test_sparse_gather_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sparse_gather = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sparse_gather)


def _gather_worker(rank: int, world: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    inputs = [
        ([1, 0, 1], [0, 2], [10.0, 12.0]),
        ([0, 1, 0], [1], [21.0]),
        ([0, 0, 0], [], []),
        ([1, 1, 1], [3, 4, 5], [33.0, 34.0, 35.0]),
    ]
    count_values, index_values, value_values = inputs[rank]

    result = sparse_gather.gather_slot_entries_to_rank0(
        torch.tensor(index_values, dtype=torch.int64),
        torch.tensor(value_values, dtype=torch.float32),
        torch.tensor(count_values, dtype=torch.int64),
        dist.group.WORLD,
    )

    if rank == 0:
        assert result is not None
        assert [part[0].tolist() for part in result] == [
            [0, 3],
            [1, 4],
            [2, 5],
        ]
        assert [part[1].tolist() for part in result] == [
            [10.0, 33.0],
            [21.0, 34.0],
            [12.0, 35.0],
        ]
    else:
        assert result is None
    dist.destroy_process_group()


def test_variable_length_p2p_gather_only_materializes_on_rank0(tmp_path) -> None:
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _gather_worker,
        args=(4, str(init_file)),
        nprocs=4,
        join=True,
    )
