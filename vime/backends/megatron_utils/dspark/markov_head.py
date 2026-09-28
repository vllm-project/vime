"""DSpark Markov head: sequential bias applied to parallel backbone logits.

Adapted from DeepSpec/deepspec/modeling/dspark/markov_head.py.

The Markov head provides a per-position bias to the draft logits, enabling
semi-autoregressive generation: the backbone produces base logits for all
positions in parallel, then the Markov head adds a bias that depends on the
previous token (teacher-forced during training, autoregressive at inference).

Three variants are supported:
    - vanilla:  W2(W1[x_{k-1}])
    - gated:    W2(gate * W1[x_{k-1}]), gate = sigmoid(Linear([h, W1[x]]))
    - rnn:      GRU-like recurrent state carrying prefix history

For vime online training, we only need the training forward (``apply_block_logits``);
sampling methods (``sample_block_tokens``) are used at inference by vLLM, not here.
"""

import torch
from torch import nn


class VanillaMarkov(nn.Module):
    """Vanilla Markov head: bias = W2(W1[prev_token])."""

    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = "vanilla"
        assert self.markov_rank > 0, f"VanillaMarkov requires markov_rank > 0, got {self.markov_rank}."
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(latent_states)

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        del hidden_states
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_step_logits(
        self,
        logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        return logits + self.compute_step_bias(token_ids, hidden_states)

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply Markov bias to all positions in a block (teacher-forced).

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids:   [B, num_blocks, block_size] (prev token per position)
            hidden_states: unused for vanilla
        Returns:
            [B, num_blocks, block_size, V]
        """
        if base_logits.size(2) == 0:
            return base_logits
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias


class GatedMarkovHead(VanillaMarkov):
    """Gated Markov head: bias = W2(gate * W1[prev_token]), gate from [h, W1[x]]."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "gated"
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_gate(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate_inputs = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return torch.sigmoid(self.gate_proj(gate_inputs))

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        prev_embeddings = self.get_prev_embeddings(token_ids)
        gate = self.compute_gate(token_ids, hidden_states).to(dtype=prev_embeddings.dtype)
        return self.project_bias(gate * prev_embeddings)


class RNNHead(VanillaMarkov):
    """RNN-based head with GRU-like recurrent state across positions in a block."""

    def __init__(
        self,
        *,
        vocab_size: int,
        markov_rank: int,
        hidden_size: int,
    ):
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.markov_head_type = "rnn"
        self.hidden_size = hidden_size
        self.state_size = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(
        self,
        state: torch.Tensor,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.project_bias(torch.tanh(output_raw))
        return new_state, bias

    def compute_step_bias(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Stateless single-step bias (state initialized to zero)."""
        assert hidden_states is not None
        prev_embeddings = self.get_prev_embeddings(token_ids)
        state = torch.zeros_like(prev_embeddings)
        _, bias = self._rnn_step(state, prev_embeddings, hidden_states)
        return bias

    def apply_block_logits(
        self,
        base_logits: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply RNN bias during training (teacher-forced, unrolled over block_size).

        Args:
            base_logits: [B, num_blocks, block_size, V]
            token_ids:   [B, num_blocks, block_size]
            hidden_states: [B, num_blocks, block_size, d]
        """
        assert hidden_states is not None
        block_size = base_logits.size(-2)
        if block_size == 0:
            return base_logits

        leading_shape = base_logits.shape[:-2]
        state = torch.zeros(
            *leading_shape,
            self.markov_rank,
            device=base_logits.device,
            dtype=hidden_states.dtype,
        )

        output_logits = []
        for k in range(block_size):
            prev_emb = self.get_prev_embeddings(token_ids[..., k])
            h_k = hidden_states[..., k, :]
            state, bias = self._rnn_step(state, prev_emb, h_k)
            output_logits.append(base_logits[..., k, :] + bias)

        return torch.stack(output_logits, dim=-2)


def build_markov_head(config) -> nn.Module | None:
    """Build a Markov head from a DSparkConfig (or compatible object)."""
    markov_rank = int(config.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank == 0:
        return None

    markov_head_type = str(config.markov_head_type).lower()
    if markov_head_type == "vanilla":
        return VanillaMarkov(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
        )
    if markov_head_type == "gated":
        return GatedMarkovHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    if markov_head_type == "rnn":
        return RNNHead(
            vocab_size=config.vocab_size,
            markov_rank=markov_rank,
            hidden_size=config.hidden_size,
        )
    raise ValueError(f"Unsupported markov_head_type: {markov_head_type!r}")
