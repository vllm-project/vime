from __future__ import annotations

from typing import Any

import torch

from vime.platforms import current_platform

from ..feature_schema import DraftFeatureSample


def collate_dspark_samples(
    samples: list[DraftFeatureSample],
    device: torch.device | str,
    *,
    block_size: int,
) -> dict[str, torch.Tensor]:
    """Pack independent feature windows into Speculators' single-sequence format."""

    if not samples:
        raise ValueError("Cannot collate an empty DSpark batch")
    if block_size < 2:
        raise ValueError("DSpark block size must be at least 2")

    tensors: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "hidden_states": [],
        "loss_mask": [],
        "verifier_last_hidden_states": [],
        "document_ids": [],
        "position_ids": [],
    }
    aux_width = int(samples[0].aux_hidden_states.size(-1))
    final_width = int(samples[0].final_hidden_states.size(-1))
    for document_id, sample in enumerate(samples):
        if sample.algorithm.lower() != "dspark":
            raise ValueError(f"DSpark collator received {sample.algorithm!r} features")
        if sample.aux_hidden_states.size(-1) != aux_width or sample.final_hidden_states.size(-1) != final_width:
            raise ValueError("DSpark batch contains inconsistent hidden sizes")
        if sample.input_ids.numel() <= block_size:
            raise ValueError("DSpark feature window is too short for one draft block")

        # Speculators' anchor selector only excludes the end of the packed tensor.
        # Mask each document tail as well so an anchor can never cross documents.
        loss_mask = sample.loss_mask.detach().clone().float()
        loss_mask[-block_size:] = 0
        tensors["input_ids"].append(sample.input_ids.long())
        tensors["hidden_states"].append(sample.aux_hidden_states)
        tensors["loss_mask"].append(loss_mask)
        tensors["verifier_last_hidden_states"].append(sample.final_hidden_states)
        tensors["document_ids"].append(torch.full_like(sample.input_ids, document_id, dtype=torch.long))
        tensors["position_ids"].append(sample.position_ids.long())

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


def dspark_trainer_kwargs(model: torch.nn.Module, args: Any) -> dict[str, Any]:
    resolver = getattr(model, "get_trainer_kwargs", None)
    if not callable(resolver):
        raise TypeError("Speculators DSpark model must implement get_trainer_kwargs()")
    loss_implementation = "eager" if current_platform().is_npu else "fused"
    train_kwargs, _ = resolver(
        loss_fn=str(args.draft_dspark_loss_fn),
        dflash_decay_gamma=float(args.draft_dspark_decay_gamma),
        max_anchors=int(args.draft_dspark_max_anchors),
        confidence_head_alpha=float(args.draft_dspark_confidence_head_alpha),
        per_position_loss_weight=str(args.draft_dspark_per_position_loss_weight),
        dpace_alpha=float(args.draft_dspark_dpace_alpha),
        loss_implementation=loss_implementation,
    )
    if loss_implementation == "eager" and "tv_loss_fn" not in train_kwargs:
        raise RuntimeError(
            "Installed Speculators does not support explicit eager DSpark losses; "
            "rebuild the NPU image from docker/Dockerfile.npu."
        )
    return train_kwargs


def compute_dspark_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    train_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the native Speculators DSpark forward and normalize metric names for VIME."""

    _, loss, raw_metrics = model(**batch, **train_kwargs)

    reference = loss.detach().new_zeros((), dtype=torch.float32)
    token_count = raw_metrics.get("full_acc_total", batch["loss_mask"].sum())
    top1_correct = raw_metrics.get("full_acc_sum", reference)
    token_count = token_count.detach().float()
    top1_correct = top1_correct.detach().float()
    metrics = {
        "loss_sum": loss.detach().float() * token_count,
        "token_count": token_count,
        "top1_correct": top1_correct,
        "top5_correct": reference,
    }
    for name in (
        "accept_rate_sum",
        "accept_rate_total",
        "accept_len_sum",
        "accept_len_total",
    ):
        value = raw_metrics.get(name)
        if torch.is_tensor(value):
            metrics[name] = value.detach().float()

    # Multi-term Speculators loss configurations report each term as a
    # ``*_loss_sum`` / ``*_loss_total`` pair (for example, ce_loss and
    # tv_loss). Keep this generic so newly supported loss functions are logged
    # without an adapter change. The bare loss_sum is intentionally excluded
    # because VIME weights its total loss by valid tokens above.
    for name, value in raw_metrics.items():
        if not name.endswith("_loss_sum") or not torch.is_tensor(value):
            continue
        total_name = f"{name.removesuffix('_sum')}_total"
        total = raw_metrics.get(total_name)
        if torch.is_tensor(total):
            metrics[name] = value.detach().float()
            metrics[total_name] = total.detach().float()
    return loss, metrics


def sync_dspark_lm_heads(
    model: torch.nn.Module,
    target_weight: torch.Tensor,
    draft_to_target_rows: torch.Tensor | None,
) -> None:
    """Keep both frozen Speculators heads aligned with the current Target version."""

    selected = target_weight
    if draft_to_target_rows is not None:
        selected = selected.index_select(0, draft_to_target_rows.to(device=selected.device, dtype=torch.long))
    for name in ("lm_head", "verifier_lm_head"):
        head = getattr(model, name, None)
        weight = getattr(head, "weight", None)
        if not torch.is_tensor(weight):
            raise RuntimeError(f"Speculators DSpark model does not expose {name}.weight")
        if weight.shape != selected.shape:
            raise ValueError(
                f"Target LM Head shape {tuple(selected.shape)} does not match DSpark {name} {tuple(weight.shape)}"
            )
        with torch.no_grad():
            weight.copy_(selected.to(device=weight.device, dtype=weight.dtype))
