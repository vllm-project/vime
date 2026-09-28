"""Export DSpark weights with the names expected by vLLM."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

_SHARED_PARAM_NAMES = {"embed_tokens.weight", "lm_head.weight"}


def export_dspark_model_weights(
    model_chunks: Sequence[nn.Module],
    *,
    use_policy_embedding: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Export draft weights and strip Megatron's vocabulary padding."""
    from megatron.core.utils import unwrap_model

    policy_model = None
    for chunk in reversed(model_chunks):
        model = unwrap_model(chunk)
        if getattr(model, "draft_model", None) is not None:
            policy_model = model
            break
    if policy_model is None:
        raise RuntimeError("DSpark is enabled, but no draft model is attached to the policy")

    draft_model = policy_model.draft_model
    config = getattr(draft_model, "config", None)
    org_vocab_size = getattr(config, "org_vocab_size", 0) or 0
    padded_vocab_size = getattr(config, "vocab_size", 0) or 0

    def strip_vocab_padding(tensor: torch.Tensor) -> torch.Tensor:
        if padded_vocab_size > org_vocab_size > 0 and tensor.dim() >= 1 and tensor.shape[0] == padded_vocab_size:
            return tensor[:org_vocab_size].contiguous()
        return tensor

    hf_state = [
        (name, strip_vocab_padding(param))
        for name, param in draft_model.named_parameters()
        if name not in _SHARED_PARAM_NAMES
    ]

    embed = (
        _get_policy_embedding(policy_model)
        if use_policy_embedding
        else getattr(getattr(draft_model, "embed_tokens", None), "weight", None)
    )
    if embed is not None:
        hf_state.append(("embed_tokens.weight", strip_vocab_padding(embed)))

    lm_head = getattr(draft_model, "lm_head", None)
    if lm_head is not None:
        hf_state.append(("lm_head.weight", strip_vocab_padding(lm_head.weight)))

    return hf_state


def _get_policy_embedding(policy_model: nn.Module) -> torch.nn.Parameter | None:
    embed = getattr(policy_model, "embedding", None)
    if embed is None:
        return None
    return getattr(embed, "word_embeddings", embed).weight
