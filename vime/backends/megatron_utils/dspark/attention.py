"""DSpark dual-input attention for Megatron backend.

Adapted from DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:Qwen3DSparkAttention.

Key difference from standard attention: K and V are computed from BOTH the
draft hidden states AND the target (policy) hidden states, then concatenated:

    k = cat([k_proj(target_hidden), k_proj(draft_hidden)], dim=seq)
    v = cat([v_proj(target_hidden), v_proj(draft_hidden)], dim=seq)

This dual-input K/V is the core architectural feature of DSpark that enables
the draft model to attend to the policy's intermediate representations.

The draft model is replicated on every TP rank and uses plain ``nn.Linear``
plus SDPA because Megatron TE attention does not support dual-input K/V.
"""

import torch
import torch.nn.functional as F
from torch import nn


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to q and k.

    Args:
        q: [bsz, heads, q_len, head_dim]
        k: [bsz, kv_heads, kv_len, head_dim]
        cos: [bsz, 1, kv_len, head_dim] (covers full kv sequence)
        sin: [bsz, 1, kv_len, head_dim]
    Returns:
        q_embed, k_embed (same shapes as q, k)
    """
    q_len = q.size(-2)
    # Q only attends to the last q_len positions of the rotary table
    # (draft positions are after context positions)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate the second half of the last dim to the front (inverse concat)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class DSparkRotaryEmbedding(nn.Module):
    """Precompute rotary sin/cos for DSpark positions.

    DSpark position ids cover both context (0..seq_len-1) and draft tokens
    (anchor_pos..anchor_pos+block_size-1 per block). The rotary table must
    cover the max position id, which is max(anchor_pos) + block_size - 1.
    """

    def __init__(self, head_dim: int, rotary_base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_base = rotary_base
        # Precompute a large table; will be indexed as needed.
        # Max position ~ seq_len + num_anchors * block_size, which is bounded
        # by the model's max_position_embeddings.
        inv_freq = 1.0 / (rotary_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) for the given position ids.

        Args:
            position_ids: [bsz, seq_len] long tensor
        Returns:
            cos, sin: [bsz, 1, seq_len, head_dim] each (broadcast-ready)
        """
        # inv_freq: [head_dim/2]
        # position_ids: [bsz, seq_len]
        # freqs: [bsz, seq_len, head_dim/2]
        inv_freq = self.inv_freq.float()  # [head_dim/2]
        positions = position_ids.float()  # [bsz, seq_len]
        # Outer product per batch: [bsz, seq_len, head_dim/2]
        freqs = torch.einsum("i,bj->bji", inv_freq, positions)
        emb = torch.cat([freqs, freqs], dim=-1)  # [bsz, seq_len, head_dim]
        cos = emb.cos()
        sin = emb.sin()

        # Add head dim for broadcasting: [bsz, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return cos.to(position_ids.dtype), sin.to(position_ids.dtype)


class DSparkParallelAttention(nn.Module):
    """Dual-input attention for DSpark draft model.

    K and V are computed from both ``hidden_states`` (draft) and
    ``target_hidden_states`` (policy), then concatenated along the sequence
    dimension. Q is computed from ``hidden_states`` only.

    The projections are intentionally unsharded because the complete draft
    model runs on every TP rank.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.attention_bias = attention_bias

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        # QK-norm (Qwen3-style RMSNorm on per-head Q/K)
        self.q_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary_emb = DSparkRotaryEmbedding(head_dim, rotary_base=rotary_base)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: [bsz, q_len, hidden] draft hidden states
            target_hidden_states: [bsz, ctx_len, hidden] policy hidden states
            position_ids: [bsz, ctx_len + q_len] position ids for rotary
            attention_mask: [bsz, 1, q_len, ctx_len + q_len] bool/float mask
                (True = attend, False = masked). If float, -inf for masked.
        Returns:
            [bsz, q_len, hidden] attention output
        """
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden_states.shape[1]

        # Q from draft hidden states
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [bsz, heads, q_len, head_dim]

        # K/V from BOTH target and draft, concatenated
        k_ctx = self.k_proj(target_hidden_states)  # [bsz, ctx_len, kv_heads * head_dim]
        k_noise = self.k_proj(hidden_states)  # [bsz, q_len, kv_heads * head_dim]
        v_ctx = self.v_proj(target_hidden_states)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # [bsz, kv_heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [bsz, kv_heads, kv_len, head_dim]

        # Apply rotary embeddings
        # position_ids: [bsz, ctx_len + q_len]
        cos, sin = self.rotary_emb(position_ids)
        # cos/sin: [1, 1, seq_len, head_dim] but we need to match k's shape
        # k shape: [bsz, kv_heads, kv_len, head_dim]
        # cos shape: [1, 1, kv_len, head_dim] -> broadcast over bsz and heads
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat K/V for GQA (grouped query attention)
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # SDPA attention
        # attention_mask: [bsz, 1, q_len, kv_len]
        # SDPA expects mask where True/1 = keep, False/0 = mask out (with bool mask)
        # Or float mask where -inf = mask out
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Convert to float bias: 0 for attend, -inf for mask
                attn_bias = torch.zeros_like(attention_mask, dtype=q.dtype)
                attn_bias = attn_bias.masked_fill(~attention_mask, float("-inf"))
            else:
                attn_bias = attention_mask
        else:
            attn_bias = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )  # [bsz, heads, q_len, head_dim]

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_attention_heads * self.head_dim)
        )
        return self.o_proj(attn_output)
