"""Check sparse delta encoding and trainer-side payload aggregation.

Verify bit-exact changed values, empty deltas and uint64 checksums, then use
transport mocks to check actual HCCL peer lengths, empty peers, padded fallback
and ownership of returned buffers. Run with pytest; no real HCCL communication.
"""

from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.update_weight.delta_sync import checksum, shard_delta_indices
from vime.backends.megatron_utils.update_weight.delta_sync import sparse_gather as module


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


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["hccl", "gloo"])
def test_root_payload_lengths_and_returned_buffer_ownership(monkeypatch, backend):
    # CPU-only transport mock: do not trigger the production NPU normalization.
    monkeypatch.setattr(module, "torch", SimpleNamespace(empty=torch.empty, cat=torch.cat))
    calls = []
    counts = [[1, 0], [2, 3], [0, 1], [0, 0]]
    peer_indices = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    peer_values = peer_indices.float()

    def batch(operations):
        for operation, tensor, peer, _group in operations:
            assert operation == "recv" and peer in (11, 12)
            assert tensor.numel() == (5 if peer == 11 else 1)
            source = peer_indices if tensor.dtype == torch.int32 else peer_values
            tensor.copy_(source if peer == 11 else torch.tensor([30], dtype=tensor.dtype))
        calls.append(operations)
        return [SimpleNamespace(wait=lambda: None)]

    def gather(tensor, buffers, **kwargs):
        assert tensor.numel() == 5
        assert tensor[1:].count_nonzero() == 0
        buffers[0].copy_(tensor)
        buffers[1].copy_(peer_indices if tensor.dtype == torch.int32 else peer_values)
        buffers[2].zero_()
        buffers[2][0] = 30
        buffers[3].zero_()
        calls.append(tensor)

    monkeypatch.setattr(
        module,
        "dist",
        SimpleNamespace(
            get_rank=lambda group: 0,
            get_world_size=lambda group: 4,
            get_global_rank=lambda group, rank: 10 + rank,
            get_backend=lambda group: backend,
            irecv="recv",
            isend="send",
            P2POp=lambda *args: args,
            batch_isend_irecv=batch,
            gather=gather,
        ),
    )
    workspace = module.GatherWorkspace()
    result = module.gather_slot_entries_to_rank0(
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1.0]),
        torch.tensor([1, 0]),
        group=object(),
        workspace=workspace,
        _counts_cpu=counts,
    )
    assert result[0][0].tolist() == [1, 10, 11]
    assert result[1][0].tolist() == [20, 21, 22, 30]
    assert torch.equal(result[0][1], result[0][0].float())
    saved = [(indices.clone(), values.clone()) for indices, values in result]
    assert len(calls) == (1 if backend == "hccl" else 2)
    assert (any(key[0] == "send_indices" for key in workspace.buffers)) == (backend != "hccl")
    # Reuse scratch; previously returned slot payloads must remain independent.
    for buffer in workspace.buffers.values():
        buffer.zero_()
    for actual, expected in zip(result, saved, strict=True):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.unit
def test_empty_hccl_peer_does_not_enqueue_p2p(monkeypatch):
    monkeypatch.setattr(module, "torch", SimpleNamespace(empty=torch.empty, cat=torch.cat))
    monkeypatch.setattr(
        module,
        "dist",
        SimpleNamespace(
            get_rank=lambda group: 1,
            get_world_size=lambda group: 2,
            get_global_rank=lambda group, rank: rank,
            get_backend=lambda group: "hccl",
            batch_isend_irecv=lambda operations: pytest.fail("empty peer must not send padding"),
        ),
    )
    assert (
        module.gather_slot_entries_to_rank0(
            torch.empty(0, dtype=torch.int32),
            torch.empty(0),
            torch.tensor([0]),
            group=object(),
            _counts_cpu=[[3], [0]],
        )
        is None
    )
