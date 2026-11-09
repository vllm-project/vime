"""Differentiable Nanbeige physical layers with independent attention per loop."""

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.utils.checkpoint import checkpoint
from vllm_rlt.layers import apply_rotary_pos_emb
from vllm_rlt.models.nanbeige import NanbeigeDecoderLayer, NanbeigeForCausalLM

from vime_plugins.ouro.model import OuroMegatronModel


class NanbeigeMegatronModel(OuroMegatronModel):
    def __init__(
        self,
        config: TransformerConfig,
        native: NanbeigeForCausalLM,
        *,
        recompute: bool = False,
        role: Literal["actor", "critic"] = "actor",
    ):
        # Reuse packing, MCore ownership and the value head, not Ouro's core math.
        super().__init__(config, native, recompute=recompute, role=role, freeze_exit_gate=False)
        self.nanbeige_config = native.config

    def _layer(
        self, layer: NanbeigeDecoderLayer, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        self.block_tokens += hidden.shape[0]
        attention = layer.self_attn
        value = layer.input_layernorm(hidden)
        shape = (hidden.shape[0], -1, attention.config.head_dim)
        q, k, v = (
            projection(value).view(shape) for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attended = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(hidden.shape[0], -1)
        )
        hidden = hidden + attention.o_proj(attended)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.model.embed_tokens(tokens)
        cos, sin = self.model.rotary_emb(hidden, torch.arange(tokens.numel(), device=tokens.device))
        for _ in range(self.loop_budget):
            for layer in self.model.layers:
                forward = partial(self._layer, layer)
                # Bound shared-weight gradient temporaries to one physical layer.
                hidden = (
                    checkpoint(forward, hidden, cos, sin, use_reentrant=True)
                    if (self.recompute and torch.is_grad_enabled())
                    else forward(hidden, cos, sin)
                )
            if not self.nanbeige_config.skip_loop_final_norm:
                hidden = self.model.norm(hidden)
        return (
            checkpoint(self._readout, hidden, use_reentrant=True)
            if self.recompute and torch.is_grad_enabled()
            else self._readout(hidden)
        )

    def _traced_sequence(self, tokens: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        if not bool((depths == self.nanbeige_config.num_loops).all()):
            raise ValueError("Nanbeige requires the checkpoint's fixed loop count")
        return self._sequence(tokens)


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
    role: Literal["actor", "critic"] = "actor",
) -> NanbeigeMegatronModel:
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("Nanbeige supports one non-pipelined model chunk")
    args = get_args()
    config = core_transformer_config_from_args(args)
    native = NanbeigeForCausalLM.from_pretrained(args.hf_checkpoint, dtype=config.params_dtype)
    return NanbeigeMegatronModel(config, native, recompute=args.recompute_granularity is not None, role=role)
