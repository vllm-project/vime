"""The DSpark L1/accept-rate terms must not materialize float32 copies of the draft or target logits.

On a 4x64GB colocate node, online draft training ran out of memory in
``_compute_accept_rate_3d`` while upcasting the [batch, 512, 7, vocab] logits
to float32. The loss now upcasts one vocab chunk at a time. These tests check
that the values and gradients match a full-tensor reference, and that nothing
the size of the full logits is kept in float32 for the backward pass.
"""

import pytest
import torch

from vime.backends.megatron_utils.dspark import loss as dspark_loss
from vime.backends.megatron_utils.dspark.common import DSparkForwardOutput

NUM_GPUS = 0


def _reference_accept_rate(draft_logits, target_logits):
    with torch.no_grad():
        pa = torch.softmax(draft_logits.float(), dim=-1)
        pb = torch.softmax(target_logits.float(), dim=-1)
        return (1.0 - 0.5 * (pa - pb).abs().sum(dim=-1)).clamp(0.0, 1.0)


def _reference_l1(draft_logits, target_logits, mask):
    pa = torch.softmax(draft_logits.float(), dim=-1)
    pb = torch.softmax(target_logits.float(), dim=-1)
    l1 = (pa - pb).abs().sum(dim=-1)
    return (l1 * mask).sum(), mask.sum()


def _inputs(dtype, vocab, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (2, 5, 3, vocab)
    draft = (torch.randn(shape, generator=generator) * 3).to(dtype).requires_grad_(True)
    target = (torch.randn(shape, generator=generator) * 3).to(dtype)
    mask = torch.rand(shape[:-1], generator=generator)
    outputs = DSparkForwardOutput.__new__(DSparkForwardOutput)
    outputs.draft_logits = draft
    return outputs, target, mask


@pytest.fixture
def small_chunks(monkeypatch):
    # several chunks, the last one partial
    monkeypatch.setattr(dspark_loss, "_L1_VOCAB_CHUNK_SIZE", 7)


@pytest.mark.unit
# bf16 gradients are rounded to bf16 (eps 7.8e-3) on both sides, so they get about two ulps of slack
@pytest.mark.parametrize("dtype, tol, grad_tol", [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 2e-3, 1.6e-2)])
@pytest.mark.parametrize("vocab", [20, 21])
def test_chunked_terms_match_full_tensor_reference(small_chunks, dtype, tol, grad_tol, vocab):
    outputs, target, mask = _inputs(dtype, vocab)

    accept = dspark_loss._compute_accept_rate_3d(outputs=outputs, aligned_target_logits=target)
    torch.testing.assert_close(accept, _reference_accept_rate(outputs.draft_logits, target), atol=tol, rtol=tol)

    num, den = dspark_loss._compute_local_l1_term(outputs=outputs, aligned_target_logits=target, loss_weight_mask=mask)
    (grad,) = torch.autograd.grad(num, outputs.draft_logits)

    ref_draft = outputs.draft_logits.detach().clone().requires_grad_(True)
    ref_num, ref_den = _reference_l1(ref_draft, target, mask)
    (ref_grad,) = torch.autograd.grad(ref_num, ref_draft)

    torch.testing.assert_close(num, ref_num, atol=tol * 10, rtol=tol)
    torch.testing.assert_close(den, ref_den)
    torch.testing.assert_close(grad.float(), ref_grad.float(), atol=grad_tol, rtol=grad_tol)


@pytest.mark.unit
def test_backward_keeps_no_float32_copy_of_the_logits(small_chunks):
    outputs, target, mask = _inputs(torch.bfloat16, 21)
    full_numel = outputs.draft_logits.numel()
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tensor.numel()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        num, _ = dspark_loss._compute_local_l1_term(
            outputs=outputs, aligned_target_logits=target, loss_weight_mask=mask
        )
    num.backward()

    big_float32 = [numel for dtype, numel in saved if dtype == torch.float32 and numel >= full_numel // 2]
    assert not big_float32, f"float32 tensors of logits size saved for backward: {big_float32}"
    assert outputs.draft_logits.grad is not None
