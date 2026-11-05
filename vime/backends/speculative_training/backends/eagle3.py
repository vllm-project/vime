from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

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


def collate_eagle3_samples(samples: list[DraftFeatureSample], device: torch.device | str) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty EAGLE3 batch")
    aligned = [align_eagle3_sample(sample) for sample in samples]
    max_rows = max(item.input_ids.size(0) for item in aligned)
    aux_hidden = aligned[0].hidden_states.size(-1)
    target_hidden = aligned[0].last_hidden_states.size(-1)
    batch_size = len(aligned)

    input_ids = torch.zeros(batch_size, max_rows, dtype=torch.long, device=device)
    hidden_states = torch.zeros(batch_size, max_rows, aux_hidden, dtype=torch.bfloat16, device=device)
    last_hidden_states = torch.zeros(batch_size, max_rows, target_hidden, dtype=torch.bfloat16, device=device)
    loss_mask = torch.zeros(batch_size, max_rows, dtype=torch.float32, device=device)
    position_ids = torch.zeros(batch_size, max_rows, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_rows, dtype=torch.bool, device=device)

    for row, item in enumerate(aligned):
        length = item.input_ids.size(0)
        if item.hidden_states.size(-1) != aux_hidden or item.last_hidden_states.size(-1) != target_hidden:
            raise ValueError("EAGLE3 batch contains inconsistent hidden sizes")
        input_ids[row, :length] = item.input_ids.to(device=device, dtype=torch.long)
        hidden_states[row, :length] = item.hidden_states.to(device=device, dtype=torch.bfloat16)
        last_hidden_states[row, :length] = item.last_hidden_states.to(device=device, dtype=torch.bfloat16)
        loss_mask[row, :length] = item.loss_mask.to(device=device, dtype=torch.float32)
        position_ids[row, :length] = item.position_ids.to(device=device, dtype=torch.long)
        attention_mask[row, :length] = True

    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "last_hidden_states": last_hidden_states,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def _normalise_draft_outputs(outputs: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        position_masks = outputs.get("position_masks")
        if position_masks is None:
            position_masks = outputs.get("loss_masks")
    else:
        logits = getattr(outputs, "logits", outputs)
        position_masks = getattr(outputs, "position_masks", None)
    logits = logits if isinstance(logits, (list, tuple)) else [logits]
    if not logits or not all(torch.is_tensor(value) for value in logits):
        raise TypeError("EAGLE3 Draft model must return a logits tensor or a list of logits tensors")
    if position_masks is None:
        position_masks = [torch.ones(value.shape[:2], dtype=torch.float32, device=value.device) for value in logits]
    elif not isinstance(position_masks, (list, tuple)):
        position_masks = [position_masks]
    if len(position_masks) != len(logits):
        raise ValueError("EAGLE3 Draft output logits and position mask counts differ")
    return list(logits), [value.squeeze(-1).float() for value in position_masks]


def compute_eagle3_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    target_lm_head_weight: torch.Tensor,
    *,
    draft_to_target_ids: torch.Tensor | None = None,
    temporal_decay: float = 0.8,
    ttt_length: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute count-normalized EAGLE3 soft-label cross entropy."""

    outputs = model(
        input_ids=batch["input_ids"],
        hidden_states=batch["hidden_states"],
        loss_mask=batch["loss_mask"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        ttt_length=ttt_length,
    )
    all_logits, all_position_masks = _normalise_draft_outputs(outputs)
    target_weight = target_lm_head_weight.to(
        device=batch["last_hidden_states"].device,
        dtype=batch["last_hidden_states"].dtype,
    )
    if draft_to_target_ids is not None:
        target_weight = target_weight.index_select(
            0, draft_to_target_ids.to(device=target_weight.device, dtype=torch.long)
        )
    with torch.no_grad():
        target_logits = F.linear(batch["last_hidden_states"], target_weight)
        target_probabilities = F.softmax(target_logits.float(), dim=-1)

    # Keep a graph-connected zero so a DDP rank with no active token can still
    # participate in the same backward collectives as the other ranks.
    total_loss = all_logits[0].float().sum() * 0.0
    total_tokens = all_logits[0].new_zeros((), dtype=torch.float32)
    top1_correct = all_logits[0].new_zeros((), dtype=torch.float32)
    top5_correct = all_logits[0].new_zeros((), dtype=torch.float32)
    seq_length = all_logits[0].size(1)
    padded_target = F.pad(target_probabilities, (0, 0, 0, len(all_logits)), value=0.0)
    padded_mask = F.pad(batch["loss_mask"].float(), (0, len(all_logits)), value=0.0)

    for step, (logits, position_mask) in enumerate(zip(all_logits, all_position_masks, strict=True)):
        if logits.size(1) != seq_length:
            raise ValueError("All EAGLE3 Draft steps must return the same sequence length")
        target_p = padded_target[:, step : step + seq_length]
        active = padded_mask[:, step : step + seq_length] * position_mask[:, :seq_length]
        finite = torch.isfinite(logits).all(dim=-1) & torch.isfinite(target_p).all(dim=-1)
        valid = (active > 0) & finite
        if not valid.any():
            continue
        per_token = -(target_p * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
        total_loss = total_loss + (float(temporal_decay) ** step) * per_token[valid].sum()
        token_count = valid.float().sum()
        total_tokens = total_tokens + token_count
        with torch.no_grad():
            target_top1 = target_p.argmax(dim=-1)[valid]
            top1_correct = top1_correct + (logits.argmax(dim=-1)[valid] == target_top1).float().sum()
            topk = min(5, int(logits.size(-1)))
            draft_topk = logits.topk(topk, dim=-1).indices[valid]
            top5_correct = top5_correct + (draft_topk == target_top1.unsqueeze(-1)).any(dim=-1).float().sum()

    if total_tokens.item() <= 0:
        return total_loss, {
            "loss_sum": total_loss.detach(),
            "token_count": total_tokens.detach(),
            "top1_correct": top1_correct.detach(),
            "top5_correct": top5_correct.detach(),
        }
    loss = total_loss / total_tokens
    return loss, {
        "loss_sum": total_loss.detach(),
        "token_count": total_tokens.detach(),
        "top1_correct": top1_correct.detach(),
        "top5_correct": top5_correct.detach(),
    }
