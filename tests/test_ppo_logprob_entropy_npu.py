"""NPU TP1/TP2 mathematical reference, separate from legacy CUDA kernel parity.

Run directly on >=2 NPUs. Missing NPU/HCCL is an error, never a passing skip.
The same 12 input scenarios as the CUDA suite use an independent FP64 CPU
log_softmax/autograd oracle. Entropy is full-vocabulary even with a replay mask.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
from ppo_logprob_entropy_test_helpers import PARITY_SCENARIOS, _full_logits, _keep_mask

from vime.utils.ppo_utils import calculate_log_probs_and_entropy

NUM_GPUS = 2
# FP32 forward/backward against independent FP64 math; GPU parity tolerances
# and its legacy BF16-gradient comparison remain unchanged in the CUDA test.
ATOL = 1e-6
RTOL = 1e-6


def assert_reference_case(device, process_group, scenario):
    chunk_size, with_mask, with_entropy, entropy_has_grad = scenario
    rank = process_group.rank() if process_group is not None else 0
    size = process_group.size() if process_group is not None else 1
    full = _full_logits()
    full_mask = _keep_mask() if with_mask else None
    targets = torch.tensor([5, 0, 3, 2], dtype=torch.long)
    local = full.chunk(size, dim=-1)[rank].to(device).contiguous()
    mask = full_mask.chunk(size, dim=-1)[rank].to(device).contiguous() if with_mask else None
    logits = local.clone().requires_grad_()
    actual_lp, actual_entropy = calculate_log_probs_and_entropy(
        logits,
        targets.to(device),
        process_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
        log_prob_keep_mask=mask,
        with_entropy_grad=entropy_has_grad,
    )
    reference = full.double().requires_grad_()
    support = reference
    if with_mask:
        keep = full_mask.clone().scatter(1, targets[:, None], True)
        support = reference.masked_fill(~keep, float("-inf"))
    expected_lp = support.log_softmax(-1).gather(1, targets[:, None])
    full_logp = reference.log_softmax(-1)
    expected_entropy = -(full_logp.exp() * full_logp).sum(-1)
    torch.testing.assert_close(actual_lp.cpu().double(), expected_lp, rtol=RTOL, atol=ATOL)
    assert torch.isfinite(actual_lp).all()
    if with_entropy:
        assert actual_entropy.requires_grad == entropy_has_grad
        torch.testing.assert_close(actual_entropy.cpu().double(), expected_entropy, rtol=RTOL, atol=ATOL)
    else:
        assert actual_entropy is None

    lp_weights = torch.tensor([0.25, -0.5, 1.5, -0.75], dtype=torch.float64)
    # Verify log-probability gradients separately so an entropy gradient cannot
    # hide a broken log-probability backward (or vice versa).
    (actual_lp.squeeze(-1) * lp_weights.to(device, dtype=torch.float32)).sum().backward()
    expected_grad = torch.autograd.grad((expected_lp.squeeze(-1) * lp_weights).sum(), reference, retain_graph=True)[0]
    torch.testing.assert_close(logits.grad.cpu().double(), expected_grad.chunk(size, -1)[rank], rtol=RTOL, atol=ATOL)
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    if with_entropy and entropy_has_grad:
        # The fused backward consumes its saved softmax buffers in place;
        # use a fresh forward for the independent entropy-gradient check.
        logits = local.clone().requires_grad_()
        _, actual_entropy = calculate_log_probs_and_entropy(
            logits,
            targets.to(device),
            process_group,
            with_entropy=True,
            chunk_size=chunk_size,
            log_prob_keep_mask=mask,
            with_entropy_grad=True,
        )
        weights = torch.tensor([0.55, -0.2, 1.8, 0.4], dtype=torch.float64)
        (actual_entropy * weights.to(device, dtype=torch.float32)).sum().backward()
        expected_grad = torch.autograd.grad((expected_entropy * weights).sum(), reference)[0]
        torch.testing.assert_close(
            logits.grad.cpu().double(), expected_grad.chunk(size, -1)[rank], rtol=RTOL, atol=ATOL
        )
        assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank, size, port):
    import torch_npu  # noqa: F401 - registers NPU devices and HCCL

    torch.npu.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("hccl", rank=rank, world_size=size, timeout=timedelta(minutes=5))
    try:
        for scenario in PARITY_SCENARIOS:
            assert_reference_case(torch.device("npu", rank), dist.group.WORLD, scenario)
        print(
            f"NPU mathematical reference passed: TP{size}, rank={rank}, {len(PARITY_SCENARIOS)} scenarios", flush=True
        )
    finally:
        dist.destroy_process_group()


def execute():
    import torch_npu  # noqa: F401

    if not torch.npu.is_available() or torch.npu.device_count() < NUM_GPUS:
        raise RuntimeError("NPU mathematical reference requires at least two visible NPUs and HCCL")
    _worker(0, 1, _free_port())
    torch.multiprocessing.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    execute()
