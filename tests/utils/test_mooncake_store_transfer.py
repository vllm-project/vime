import types

import numpy as np
import torch

from vime.utils.mooncake_store_service import ensure_mooncake_master
from vime.utils.remote_batch import MooncakeRemoteBatch, create_mooncake_store, normalize_store_init_kwargs
from vime.utils.rollout_store_transfer import rollout_store_batch_to_data, split_rollout_data_by_dp_mooncake_store


def test_mooncake_remote_batch_roundtrip():
    ensure_mooncake_master()
    store = create_mooncake_store()
    remote = MooncakeRemoteBatch.from_tensors(
        {"tokens": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)},
        store,
        prefix="vime-test/roundtrip",
    )
    try:
        assert remote.materialize()["tokens"].tolist() == [[1, 2, 3], [4, 5, 6]]
    finally:
        remote.cleanup()


def test_split_and_materialize_roundtrip():
    ensure_mooncake_master()
    args = types.SimpleNamespace(transfer_backend="mooncake_store", mooncake_store_init_kwargs=None)
    data = {
        "tokens": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
        "loss_masks": [[1, 1, 1], [1, 1], [1, 1, 1, 1]],
        "response_lengths": [3, 2, 4],
        "rewards": [0.1, 0.2, 0.3],
        "truncated": [0, 0, 1],
        "rollout_ids": [0, 0, 1],
        "total_lengths": [3, 2, 4],
        "global_batch_sizes": [2, 1],
        "num_microbatches": 2,
    }
    refs = split_rollout_data_by_dp_mooncake_store(
        args,
        data,
        dp_size=2,
        partitions=[[0, 1], [2]],
        micro_batch_indices=[[0, 1], [0]],
        num_microbatches=2,
        global_batch_sizes=[2, 1],
    )
    assert len(refs) == 2
    rollout = rollout_store_batch_to_data(refs[0])
    assert rollout["partition"] == [0, 1]
    assert all(isinstance(row, torch.Tensor) for row in rollout["tokens"])
    assert rollout["tokens"][0].tolist() == [1, 2, 3]


def test_normalize_store_init_kwargs_uses_env_defaults():
    kwargs = normalize_store_init_kwargs(None)
    assert kwargs["protocol"] == "tcp"
    assert kwargs["master_server_addr"] == "127.0.0.1:50051"
