"""CPU DSpark loss regressions; set VIME_TEST_CUDA=1 to also exercise CUDA."""

import os
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vime.backends.megatron_utils.dspark import loss as dspark_loss
from vime.backends.megatron_utils.dspark.common import DSparkConfig, DSparkForwardOutput

NUM_GPUS = 0
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            os.environ.get("VIME_TEST_CUDA") != "1" or not torch.cuda.is_available(),
            reason="optional CUDA coverage requires VIME_TEST_CUDA=1 and a CUDA device",
        ),
    ),
]


def _outputs(device="cpu", dtype=torch.float32, vocab_size=19, mask="partial", confidence=True, target=True):
    generator = torch.Generator().manual_seed(123)
    shape = (2, 2, 3)

    def leaf(shape):
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype).requires_grad_()

    draft_logits = leaf((*shape, vocab_size))
    target_logits = leaf((*shape, vocab_size)) if target else None
    confidence_pred = leaf(shape) if confidence else None
    eval_mask = torch.ones(shape, dtype=torch.bool, device=device)
    if mask == "zero":
        eval_mask.zero_()
    elif mask == "partial":
        eval_mask[0, 0, 1:] = False
        eval_mask[1, 1] = False
    return DSparkForwardOutput(
        draft_logits=draft_logits,
        target_ids=torch.randint(vocab_size, shape, generator=generator).to(device),
        eval_mask=eval_mask,
        block_keep_mask=eval_mask.any(dim=-1),
        confidence_pred=confidence_pred,
        aligned_target_logits=target_logits,
    )


def _clone_outputs(outputs):
    return replace(
        outputs,
        **{
            name: value.detach().clone().requires_grad_() if value is not None else None
            for name in ("draft_logits", "aligned_target_logits", "confidence_pred")
            for value in [getattr(outputs, name)]
        },
    )


def _dense_reference(outputs, config, world_size=1, extra_denominators=(0.0, 0.0, 0.0)):
    """Independent full-softmax formula, with no production loss helpers."""
    logits = outputs.draft_logits
    weights = outputs.eval_mask.float()
    if config.loss_decay_gamma is not None and config.loss_decay_gamma > 0:
        positions = torch.arange(logits.shape[-2], device=logits.device)
        weights = weights * torch.exp(-positions.float() / config.loss_decay_gamma)
    ce = F.cross_entropy(logits.flatten(0, 2), outputs.target_ids.flatten(), reduction="none")
    nums = [(ce.reshape_as(weights) * weights).sum()]
    dens = [weights.sum()]
    distance = None
    if config.l1_loss_alpha > 0 or outputs.confidence_pred is not None:
        distance = (
            (logits.float().softmax(dim=-1) - outputs.aligned_target_logits.float().softmax(dim=-1)).abs().sum(dim=-1)
        )
    zero = nums[0].new_zeros(())
    nums.append((distance * weights).sum() if config.l1_loss_alpha > 0 else zero)
    dens.append(weights.sum() if config.l1_loss_alpha > 0 else zero)
    if outputs.confidence_pred is not None:
        acceptance = (1.0 - distance.detach() / 2.0).clamp(0.0, 1.0)
        conf = F.binary_cross_entropy_with_logits(outputs.confidence_pred.float(), acceptance, reduction="none")
        nums.append((conf * weights).sum())
        dens.append(weights.sum())
    else:
        nums.append(zero)
        dens.append(zero)
    local = [num / (den + 1e-6) for num, den in zip(nums, dens, strict=True)]
    # Preserve the existing zero-denominator branch: an all-masked L1 term has no gradient edge.
    if dens[1].item() == 0:
        local[1] = zero
    alphas = (config.ce_loss_alpha, config.l1_loss_alpha, config.confidence_head_alpha)
    global_losses = [
        num / (den.detach() * world_size + extra + 1e-6)
        for num, den, extra in zip(nums, dens, extra_denominators, strict=True)
    ]
    if (dens[1] * world_size + extra_denominators[1]).item() == 0:
        global_losses[1] = zero
    backward = sum(alpha * term for alpha, term in zip(alphas, global_losses, strict=True)) * world_size
    metrics = {
        f"dspark/{name}_loss": value.detach().item()
        for name, value in zip(("ce", "l1", "confidence"), local, strict=True)
    }
    metrics["dspark/total_loss"] = sum(alpha * term for alpha, term in zip(alphas, local, strict=True)).detach().item()
    return backward, metrics


def _assert_matches_reference(outputs, config, **reference_kwargs):
    reference_outputs = _clone_outputs(outputs)
    actual, actual_metrics = dspark_loss.compute_dspark_loss(outputs=outputs, config=config)
    expected, expected_metrics = _dense_reference(reference_outputs, config, **reference_kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert actual_metrics == pytest.approx(expected_metrics, rel=2e-5, abs=2e-6)
    actual.backward()
    expected.backward()
    for name in ("draft_logits", "aligned_target_logits", "confidence_pred"):
        value, reference = getattr(outputs, name), getattr(reference_outputs, name)
        if value is None:
            continue
        if reference.grad is None:
            assert value.grad is None, name
        else:
            assert value.grad is not None, name
            rtol, atol = (2e-2, 2e-7) if value.dtype == torch.bfloat16 else (2e-5, 2e-7)
            torch.testing.assert_close(value.grad, reference.grad, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("l1_alpha,confidence", [(0.0, False), (0.0, True), (0.9, False), (0.9, True)])
@pytest.mark.parametrize("mask", ["full", "partial", "zero"])
@pytest.mark.parametrize("decay", [None, 4.0, 0.0, -1.0])
def test_loss_metrics_and_gradients_match_dense_reference(
    device, dtype, l1_alpha, confidence, mask, decay, monkeypatch
):
    monkeypatch.setattr(dspark_loss, "_L1_VOCAB_CHUNK_SIZE", 8)
    outputs = _outputs(device, dtype, mask=mask, confidence=confidence)
    config = DSparkConfig(l1_loss_alpha=l1_alpha, loss_decay_gamma=decay, confidence_head_alpha=0.7)
    _assert_matches_reference(outputs, config)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_production_vocab_chunk_boundaries(device, dtype, offset):
    outputs = _outputs(device, dtype, vocab_size=dspark_loss._L1_VOCAB_CHUNK_SIZE + offset)
    _assert_matches_reference(outputs, DSparkConfig())


@pytest.mark.parametrize(
    "draft_requires_grad,target_requires_grad", [(False, False), (True, False), (False, True), (True, True)]
)
def test_checkpoint_requires_a_trainable_input(draft_requires_grad, target_requires_grad, monkeypatch):
    monkeypatch.setattr(dspark_loss, "_L1_VOCAB_CHUNK_SIZE", 8)
    outputs = _outputs()
    draft = outputs.draft_logits.requires_grad_(draft_requires_grad)
    target = outputs.aligned_target_logits.requires_grad_(target_requires_grad)
    reference_draft = draft.detach().clone().requires_grad_(draft_requires_grad)
    reference_target = target.detach().clone().requires_grad_(target_requires_grad)
    with torch.enable_grad(), patch.object(
        dspark_loss.checkpoint, "checkpoint", wraps=dspark_loss.checkpoint.checkpoint
    ) as checkpoint_spy:
        actual = dspark_loss._compute_probability_l1(draft_logits=draft, target_logits=target)
        expected = (reference_draft.softmax(dim=-1) - reference_target.softmax(dim=-1)).abs().sum(dim=-1)
    needs_grad = draft_requires_grad or target_requires_grad
    assert checkpoint_spy.call_count == (3 if needs_grad else 0)
    assert actual.requires_grad == needs_grad
    torch.testing.assert_close(actual, expected)
    if needs_grad:
        actual.sum().backward()
        expected.sum().backward()
        for tensor, reference in ((draft, reference_draft), (target, reference_target)):
            if tensor.requires_grad:
                torch.testing.assert_close(tensor.grad, reference.grad, rtol=2e-5, atol=2e-7)
            else:
                assert tensor.grad is None


@pytest.mark.parametrize("target", [False, True])
def test_ce_only_without_target_or_probability_work(target):
    outputs = _outputs(confidence=False, target=target)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        dspark_loss.compute_dspark_loss(outputs=outputs, config=DSparkConfig(l1_loss_alpha=0.0))
    assert not any(event.name == "aten::logsumexp" for event in profile.events())
    _assert_matches_reference(outputs, DSparkConfig(l1_loss_alpha=0.0))


@pytest.mark.parametrize("l1_alpha,confidence", [(0.0, True), (0.9, False), (0.9, True)])
def test_probability_normalizers_are_computed_once(l1_alpha, confidence):
    outputs = _outputs(confidence=confidence)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        dspark_loss.compute_dspark_loss(outputs=outputs, config=DSparkConfig(l1_loss_alpha=l1_alpha))
    # The out= overload is a nested profiler event; count only top-level calls.
    normalizers = [event for event in profile.events() if event.name == "aten::logsumexp" and event.cpu_parent is None]
    assert len(normalizers) == 2


@pytest.mark.parametrize(
    "l1_alpha,confidence,message",
    [
        (0.9, False, "aligned_target_logits is required when l1_loss_alpha > 0"),
        (0.9, True, "aligned_target_logits is required when l1_loss_alpha > 0"),
        (0.0, True, "aligned_target_logits is required when confidence head is enabled"),
    ],
)
def test_missing_required_target(l1_alpha, confidence, message):
    with pytest.raises(AssertionError, match=message):
        dspark_loss.compute_dspark_loss(
            outputs=_outputs(confidence=confidence, target=False), config=DSparkConfig(l1_loss_alpha=l1_alpha)
        )


@pytest.mark.parametrize("l1_alpha", [0.0, 0.9])
def test_confidence_term_cannot_update_draft_or_target_logits(l1_alpha):
    outputs = _outputs()
    terms, _ = dspark_loss._collect_local_terms(outputs=outputs, loss_decay_gamma=4.0, l1_loss_alpha=l1_alpha)
    terms["confidence_loss_num"].backward()
    assert outputs.draft_logits.grad is None
    assert outputs.aligned_target_logits.grad is None
    assert outputs.confidence_pred.grad is not None
    assert outputs.confidence_pred.grad.abs().sum().item() > 0


def test_confidence_only_does_not_save_target_for_backward():
    outputs = _outputs()
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        dspark_loss.compute_dspark_loss(outputs=outputs, config=DSparkConfig(l1_loss_alpha=0.0))
    assert all(tensor.data_ptr() != outputs.aligned_target_logits.data_ptr() for tensor in saved)


@pytest.mark.parametrize("l1_alpha,confidence", [(0.0, True), (0.9, False), (0.9, True)])
def test_outer_no_grad_is_preserved(l1_alpha, confidence):
    outputs = _outputs(confidence=confidence)
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor), torch.no_grad():
        terms, _ = dspark_loss._collect_local_terms(outputs=outputs, loss_decay_gamma=4.0, l1_loss_alpha=l1_alpha)
        actual, metrics = dspark_loss.compute_dspark_loss(outputs=outputs, config=DSparkConfig(l1_loss_alpha=l1_alpha))
        expected, expected_metrics = _dense_reference(outputs, DSparkConfig(l1_loss_alpha=l1_alpha))
    assert all(not value.requires_grad for value in terms.values())
    assert not saved
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    assert metrics == pytest.approx(expected_metrics)


def test_global_denominators_and_world_size_scaling(monkeypatch):
    world_size = 3
    extra = (2.0, 3.0, 4.0)
    monkeypatch.setattr(dspark_loss.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dspark_loss.dist, "get_world_size", lambda: world_size)

    def denominators(terms, *, world_size):
        return {
            key: terms[key].detach() * world_size + extra[index]
            for index, key in enumerate(("ce_loss_den", "l1_loss_den", "confidence_loss_den"))
        }

    monkeypatch.setattr(dspark_loss, "_all_reduce_loss_denominators", denominators)
    _assert_matches_reference(_outputs(), DSparkConfig(), world_size=world_size, extra_denominators=extra)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
