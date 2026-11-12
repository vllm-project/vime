from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vime.platforms import current_platform

from ..feature_schema import DraftFeatureSample


@dataclass(slots=True)
class Eagle3AlignedSample:
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    last_hidden_states: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor


def align_eagle3_sample(sample: DraftFeatureSample) -> Eagle3AlignedSample:
    """Apply the EAGLE3 p/p+1/p+2 training alignment."""

    sample.validate(strict=True)
    train_rows = int(sample.input_ids.numel()) - 2
    if train_rows <= 0:
        raise ValueError("EAGLE3 sample has no trainable rows after future-token alignment")
    return Eagle3AlignedSample(
        input_ids=sample.input_ids[1 : 1 + train_rows],
        hidden_states=sample.aux_hidden_states[:train_rows],
        last_hidden_states=sample.final_hidden_states[1 : 1 + train_rows],
        loss_mask=sample.loss_mask[2 : 2 + train_rows],
        position_ids=sample.position_ids[:train_rows],
    )


def collate_eagle3_samples(
    samples: list[DraftFeatureSample],
    device: torch.device | str,
    *,
    ttt_steps: int = 1,
) -> dict[str, torch.Tensor]:
    """Pack aligned feature windows into the Speculators single-sequence format."""

    if not samples:
        raise ValueError("Cannot collate an empty EAGLE3 batch")
    if ttt_steps < 1:
        raise ValueError("EAGLE3 TTT steps must be at least 1")

    tensors: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "hidden_states": [],
        "loss_mask": [],
        "verifier_last_hidden_states": [],
        "document_ids": [],
        "position_ids": [],
    }
    aux_width = int(samples[0].aux_hidden_states.size(-1))
    target_width = int(samples[0].final_hidden_states.size(-1))
    for document_id, sample in enumerate(samples):
        if sample.algorithm.lower() != "eagle3":
            raise ValueError(f"EAGLE3 collator received {sample.algorithm!r} features")
        if (
            int(sample.aux_hidden_states.size(-1)) != aux_width
            or int(sample.final_hidden_states.size(-1)) != target_width
        ):
            raise ValueError("EAGLE3 batch contains inconsistent hidden sizes")
        aligned = align_eagle3_sample(sample)
        loss_mask = aligned.loss_mask.detach().clone().float()
        # Speculators pairs row i of TTT step t with the Target row i+t.  A
        # pairing that crosses a document boundary would train a row on a
        # foreign (token, target) pair, so drop the leading ttt_steps-1 rows
        # of every document except the first from the loss; rows the packed
        # tail cannot reach are already excluded by the model itself.
        if ttt_steps > 1 and document_id > 0:
            loss_mask[: ttt_steps - 1] = 0
        tensors["input_ids"].append(aligned.input_ids.to(torch.long))
        tensors["hidden_states"].append(aligned.hidden_states)
        tensors["loss_mask"].append(loss_mask)
        tensors["verifier_last_hidden_states"].append(aligned.last_hidden_states)
        tensors["document_ids"].append(
            torch.full_like(aligned.input_ids.to(torch.long), document_id, dtype=torch.long)
        )
        tensors["position_ids"].append(aligned.position_ids.to(torch.long))

    def packed(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.cat(tensors[name], dim=0).to(device=device, dtype=dtype).unsqueeze(0).contiguous()

    return {
        "input_ids": packed("input_ids", torch.long),
        "hidden_states": packed("hidden_states", torch.bfloat16),
        "loss_mask": packed("loss_mask", torch.float32),
        "verifier_last_hidden_states": packed("verifier_last_hidden_states", torch.bfloat16),
        "document_ids": packed("document_ids", torch.long),
        "position_ids": packed("position_ids", torch.long),
    }


def eagle3_trainer_kwargs(model: torch.nn.Module, args: Any) -> dict[str, Any]:
    """Resolve the Speculators EAGLE3 training configuration from CLI arguments."""

    resolver = getattr(model, "get_trainer_kwargs", None)
    if not callable(resolver):
        raise TypeError("The Speculators EAGLE3 model must implement get_trainer_kwargs()")
    # The fused loss implementation needs Triton kernels that do not run on
    # Ascend.
    loss_implementation = "eager" if current_platform().is_npu else "fused"
    train_kwargs, _ = resolver(
        loss_fn=str(args.draft_eagle3_loss_fn),
        loss_implementation=loss_implementation,
        ttt_steps=int(args.draft_ttt_length),
        ttt_step_loss_decay=float(args.draft_temporal_decay),
    )
    return train_kwargs


def compute_eagle3_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    train_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the native Speculators EAGLE3 forward and adapt its metrics for VIME."""

    _, loss, raw_metrics = model(**batch, **train_kwargs)

    token_count = None
    top1_correct = None
    ttt_steps = int(train_kwargs.get("ttt_steps", 1))
    for step in range(ttt_steps):
        total = raw_metrics.get(f"full_acc_{step}_total")
        correct = raw_metrics.get(f"full_acc_{step}_sum")
        if torch.is_tensor(total):
            total = total.detach().float()
            token_count = total if token_count is None else token_count + total
        if torch.is_tensor(correct):
            correct = correct.detach().float()
            top1_correct = correct if top1_correct is None else top1_correct + correct
    if token_count is None:
        # Metric-free models still need a token count for the DDP-average.
        token_count = batch["loss_mask"].sum().detach().float()
    if top1_correct is None:
        top1_correct = loss.detach().new_zeros((), dtype=torch.float32)
    metrics = {
        "loss_sum": loss.detach().float() * token_count,
        "token_count": token_count,
        "top1_correct": top1_correct,
        "top5_correct": loss.detach().new_zeros((), dtype=torch.float32),
    }
    return loss, metrics
