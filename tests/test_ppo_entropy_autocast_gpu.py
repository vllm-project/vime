"""Single-GPU autocast regressions for the fused PPO entropy reduction."""

from __future__ import annotations

import math

import pytest
import torch

from vime.utils.ppo_utils import calculate_log_probs_and_entropy

NUM_GPUS = 1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def autocast_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    dtype = request.param
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 autocast requires a supported CUDA device")
    return dtype


def test_uniform_entropy_is_shift_invariant_under_cuda_autocast(autocast_dtype: torch.dtype):
    # The offset is not representable in fp16 or bf16, but all logits are fp32.
    # An autocast dot product must not turn this shift into an entropy gradient.
    logits = torch.zeros((2, 256), dtype=torch.float32, device="cuda")
    logits[1].fill_(100.03125)
    logits.requires_grad_()
    tokens = torch.tensor([0, 255], dtype=torch.long, device="cuda")

    with torch.autocast("cuda", dtype=autocast_dtype):
        _, entropy = calculate_log_probs_and_entropy(logits, tokens, None, with_entropy=True)

    assert entropy is not None
    torch.testing.assert_close(entropy, torch.full_like(entropy, math.log(256)), rtol=0, atol=1e-5)
    entropy.sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=1e-7)


@pytest.mark.parametrize("with_mask", [False, True], ids=["unmasked", "masked"])
@pytest.mark.parametrize("with_entropy_grad", [False, True], ids=["metric_only", "entropy_grad"])
@pytest.mark.parametrize("chunk_size", [-1, 2], ids=["no_chunks", "chunks"])
def test_cuda_autocast_entropy_matches_fp32_forward_and_backward(
    autocast_dtype: torch.dtype,
    with_mask: bool,
    with_entropy_grad: bool,
    chunk_size: int,
):
    generator = torch.Generator().manual_seed(37)
    initial_logits = torch.randn((5, 257), generator=generator, dtype=torch.float32).to("cuda")
    logits = initial_logits.clone().requires_grad_()
    reference_logits = initial_logits.clone().requires_grad_()
    tokens = torch.tensor([1, 16, 37, 128, 256], device="cuda")
    keep_mask = None
    if with_mask:
        keep_mask = ((torch.arange(257, device="cuda") % 3) == 0).expand(5, -1)

    kwargs = {
        "tp_group": None,
        "with_entropy": True,
        "with_entropy_grad": with_entropy_grad,
        "chunk_size": chunk_size,
        "log_prob_keep_mask": keep_mask,
    }
    with torch.autocast("cuda", enabled=False):
        reference_log_probs, reference_entropy = calculate_log_probs_and_entropy(reference_logits, tokens, **kwargs)
    with torch.autocast("cuda", dtype=autocast_dtype):
        log_probs, entropy = calculate_log_probs_and_entropy(logits, tokens, **kwargs)
        assert torch.is_autocast_enabled()

    assert entropy is not None and reference_entropy is not None
    assert entropy.dtype == torch.float32
    assert entropy.requires_grad == with_entropy_grad
    torch.testing.assert_close(log_probs, reference_log_probs, rtol=0, atol=1e-6)
    torch.testing.assert_close(entropy, reference_entropy, rtol=0, atol=1e-6)

    weights = torch.tensor([0.5, -0.25, 1.0, -0.75, 1.25], device="cuda")
    loss = (log_probs.squeeze(-1) * weights).sum()
    reference_loss = (reference_log_probs.squeeze(-1) * weights).sum()
    if with_entropy_grad:
        loss = loss + (entropy * weights).sum()
        reference_loss = reference_loss + (reference_entropy * weights).sum()
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0, atol=1e-7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
