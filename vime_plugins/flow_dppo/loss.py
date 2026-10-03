"""Full-vocabulary categorical KL; this changes the Gaussian policy family."""

import torch
import torch.nn.functional as F


def categorical_flow_dppo(
    logits: torch.Tensor,
    old_log_probs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    divergence_budget: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    old_log_probs = old_log_probs.detach().to(log_probs.device)
    divergence = (old_log_probs.exp() * (old_log_probs - log_probs)).sum(-1).clamp_min(0)
    log_ratio = (log_probs - old_log_probs).gather(-1, actions[:, None])[:, 0]
    ratio = log_ratio.exp()
    outward = ((advantages > 0) & (log_ratio > 0)) | ((advantages < 0) & (log_ratio < 0))
    blocked = outward & (divergence.detach() > divergence_budget)
    loss = -torch.where(blocked, ratio.detach(), ratio) * advantages
    return loss, blocked.float(), divergence.detach()
