"""Process-local tests adapted from verl's sharded delta suite."""

import pytest
import torch

from vime.backends.megatron_utils.update_weight.delta_sync import checksum, shard_delta_indices


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_delta_indices_is_bit_exact(dtype):
    torch.manual_seed(0)
    previous = torch.randn(1000, dtype=dtype)
    current = previous.clone()
    changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64)
    current[changed] += 0.5

    indices, values = shard_delta_indices(current, previous, offset=4096)

    assert torch.equal(indices, changed + 4096)
    integer_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(values.view(integer_dtype), current[changed].view(integer_dtype))


@pytest.mark.unit
def test_shard_delta_indices_no_change_is_empty():
    previous = torch.randn(256, dtype=torch.bfloat16)
    indices, values = shard_delta_indices(previous.clone(), previous, offset=0)
    assert indices.numel() == 0
    assert values.numel() == 0


@pytest.mark.unit
def test_checksum_fits_uint64_wire_format(monkeypatch):
    hashes = iter([2**63 - 1, 2**63 - 1])
    monkeypatch.setattr(torch, "hash_tensor", lambda _: torch.tensor(next(hashes)))

    result = checksum(torch.ones(1, dtype=torch.uint8), torch.ones(1))

    assert 0 <= result <= 2**64 - 1
