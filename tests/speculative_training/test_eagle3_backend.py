from __future__ import annotations

import pytest
import torch

from vime.backends.speculative_training.backends.eagle3 import (
    align_eagle3_sample,
    collate_eagle3_samples,
    compute_eagle3_loss,
)
from vime.backends.speculative_training.feature_schema import DraftFeatureSample


def _sample() -> DraftFeatureSample:
    rows = 6
    aux = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    final = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    return DraftFeatureSample(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]),
        loss_mask=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32),
        position_ids=torch.arange(20, 20 + rows),
        aux_hidden_states=aux,
        final_hidden_states=final,
        rollout_id=0,
        target_weight_version="1",
        original_sample_id="s0",
        prompt_length=2,
        response_length=4,
        window_start=0,
        window_end=rows,
        aux_layer_ids=(1,),
    )


@pytest.mark.unit
def test_eagle3_alignment_is_p_p1_p2():
    sample = _sample()
    aligned = align_eagle3_sample(sample)

    assert aligned.input_ids.tolist() == [11, 12, 13, 14]
    assert torch.equal(aligned.hidden_states, sample.aux_hidden_states[:4])
    assert torch.equal(aligned.last_hidden_states, sample.final_hidden_states[1:5])
    assert aligned.loss_mask.tolist() == [1, 1, 1, 1]
    assert aligned.position_ids.tolist() == [20, 21, 22, 23]


class _PerfectDraft(torch.nn.Module):
    def __init__(self, target_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target_weight", target_weight)

    def forward(self, *, last_hidden_states=None, **kwargs):
        del last_hidden_states
        logits = torch.nn.functional.linear(kwargs["hidden_states"][..., :3], self.target_weight)
        return {"logits": [logits], "position_masks": [kwargs["loss_mask"]]}


@pytest.mark.unit
def test_eagle3_loss_is_finite_and_count_normalized():
    batch = collate_eagle3_samples([_sample()], device="cpu")
    # Make aux[:3] equal the already-aligned final hidden so the simple Draft has
    # the same ranking as the Target head. The test exercises the loss contract,
    # not a particular Draft architecture.
    batch["hidden_states"][..., :3] = batch["last_hidden_states"]
    target_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    model = _PerfectDraft(target_weight)

    loss, metrics = compute_eagle3_loss(model, batch, target_weight)

    assert torch.isfinite(loss)
    assert metrics["token_count"].item() == 4
    assert metrics["top1_correct"].item() == 4
    assert metrics["top5_correct"].item() == 4


@pytest.mark.unit
def test_eagle3_loss_supports_restricted_draft_vocab():
    batch = collate_eagle3_samples([_sample()], device="cpu")
    target_weight = torch.randn(8, 3, dtype=torch.bfloat16)
    draft_ids = torch.tensor([1, 3, 6])

    class RestrictedDraft(torch.nn.Module):
        def forward(self, **kwargs):
            shape = (*kwargs["input_ids"].shape, 3)
            return {"logits": torch.zeros(shape), "position_masks": kwargs["loss_mask"]}

    loss, metrics = compute_eagle3_loss(
        RestrictedDraft(),
        batch,
        target_weight,
        draft_to_target_ids=draft_ids,
    )

    assert torch.isfinite(loss)
    assert metrics["token_count"].item() == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
