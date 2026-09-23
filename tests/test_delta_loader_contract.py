"""CPU contract tests for the actual Ascend loader and delta exporter."""

from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.update_weight.megatron_delta_export import _device_shard
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import _verify_dense_load
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SparseWeightPatch,
    apply_sparse_hf_patches_with_loader,
)


class Model(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)[rank * 2:rank * 2 + 2].clone())
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, weight in dict(weights).items():
            assert name == "weight"
            self.weight.data.copy_(weight[self.rank * 2:self.rank * 2 + 2])


def test_cpu_backup_cannot_silently_become_diff_source():
    with pytest.raises(RuntimeError, match="live accelerator shard"):
        _device_shard(SimpleNamespace(param=torch.ones(4), megatron_name="weight"))


@pytest.mark.parametrize("rank", [0, 1])
def test_three_real_deltas_split_and_empty_preserve_full_state(rank):
    model = Model(rank)
    expected = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    for round_id in range(3):
        indices = torch.tensor([round_id, 7 - round_id], dtype=torch.int32)
        values = torch.tensor([20 + round_id, 40 + round_id], dtype=torch.bfloat16)
        # Same HF name appears in two wire pieces. Both must be applied.
        patches = [(SparseWeightPatch("weight", indices[i:i+1], values[i:i+1]), [4, 2]) for i in range(2)]
        apply_sparse_hf_patches_with_loader(model, patches, chunk_bytes=16)
        expected.view(-1).index_copy_(0, indices.long(), values)
        assert torch.equal(model.weight.view(torch.int16), expected[rank*2:rank*2+2].view(torch.int16))
        _verify_dense_load(model, [("weight", expected)])
    apply_sparse_hf_patches_with_loader(model, [])


def test_bad_wire_bounds_fail_before_mutation():
    model = Model(0)
    before = model.weight.detach().clone()
    patches = [(SparseWeightPatch("weight", torch.tensor([8], dtype=torch.int32), torch.ones(1)), [4, 2])]
    with pytest.raises(IndexError):
        apply_sparse_hf_patches_with_loader(model, patches)
    assert torch.equal(model.weight, before)


def test_dense_verify_detects_missing_update():
    model = Model(0)
    with pytest.raises(RuntimeError, match="state verification failed"):
        _verify_dense_load(model, [("weight", torch.ones(4, 2, dtype=torch.bfloat16))])
