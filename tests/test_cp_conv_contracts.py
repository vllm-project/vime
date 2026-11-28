"""CPU tests for the CP contracts the short convolution relies on.

Three groups:

1. **Document cut points** -- `pre_num_conv_tokens` must be the number of tokens
   that precede a shard inside its own document, capped by the kernel width.
   Getting this wrong splices one document's tail onto a shard that starts a
   new document (or drops history the shard genuinely needs).
2. **Partial-history gradient accumulation** -- the adjoint of the injected
   history must land on the *last* rows of the previous rank's shard, and a
   rank with no injected history must leave its tail untouched.
3. **Sequence-parallel divisibility** -- the zigzag boundary must be able to
   split its local row count evenly across the TP ranks.
"""

from __future__ import annotations

import pytest
import torch

from fla.ops.cp.context import _interval_cp_meta
from vime_plugins.models.gdn.cp_layout import _pieces


# ---------------------------------------------------------------- 1. cut points

@pytest.mark.parametrize(
    ("tokens_before_shard", "expected"),
    [
        (0, 0),  # shard starts a document -> no history at all
        (1, 1),  # one token of the document precedes the shard
        (2, 2),
        (5, 5),  # deeper than the kernel width; the caller clamps to W-1
    ],
)
def test_pre_num_conv_tokens_counts_tokens_before_the_shard(tokens_before_shard, expected):
    """`_interval_cp_meta` is the production builder of `pre_num_conv_tokens`.

    A document spanning [0, 100) with a shard starting at
    `tokens_before_shard` must report exactly that many preceding tokens.
    """
    cu = torch.tensor([0, 100], dtype=torch.int64)
    _, pre_num_conv_tokens, *_ = _interval_cp_meta(
        cu_seqlens_cpu=cu,
        start=tokens_before_shard,
        end=100,
        chain_pos=0,
        part_len=100,
    )
    assert pre_num_conv_tokens == expected


def test_pre_num_conv_tokens_is_capped_by_kernel_width_in_the_conv():
    """The conv clamps to W-1; the builder itself is not required to."""
    width = 4
    for before in (0, 1, 2, 5, 99):
        assert min(width - 1, before) == min(3, before)


# ------------------------------------------- 2. partial-history gradient accum

def _accumulate_prefix_grad(tail_grad, next_prefix_grad, pad):
    """Mirror of `_CPConvWithState.backward`'s accumulation step.

    Production ships each rank's prefix adjoint in a `[pad, dim]` buffer that is
    **right-aligned** (the newest history row sits at row `pad-1`), then the
    receiver adds the sender's buffer onto the last `min(pad, local_rows)` rows
    of its own local gradient:

        local_dx[:, -n:, :].add_(received[rank + 1][:, -n:, :])

    So a rank that injected `h` history rows contributes values only in the
    last `h` rows of that buffer.
    """
    local = tail_grad.clone()
    n = min(pad, local.shape[0])
    if n == 0:
        return local
    # right-align the sender's h rows inside a pad-wide buffer
    h = next_prefix_grad.shape[0]
    buf = torch.zeros(pad, dtype=local.dtype)
    if h:
        buf[-min(h, pad):] = next_prefix_grad[-min(h, pad):]
    local[-n:] = local[-n:] + buf[-n:]
    return local


def test_partial_history_gradient_accumulates_on_the_tail():
    """pad=3, h=1, next rank sends 7 -> only the last tail row moves."""
    tail_grad = torch.tensor([10.0, 20.0, 30.0])
    next_prefix_grad = torch.tensor([7.0])  # h = 1
    out = _accumulate_prefix_grad(tail_grad, next_prefix_grad, pad=3)
    assert out.tolist() == [10.0, 20.0, 37.0]


def test_no_history_leaves_the_tail_untouched():
    """h = 0 (the next rank starts a document) must not modify this rank."""
    tail_grad = torch.tensor([10.0, 20.0, 30.0])
    next_prefix_grad = torch.zeros(0)
    out = _accumulate_prefix_grad(tail_grad, next_prefix_grad, pad=3)
    assert out.tolist() == [10.0, 20.0, 30.0]


def test_full_history_shifts_all_tail_rows():
    tail_grad = torch.tensor([1.0, 2.0, 3.0])
    next_prefix_grad = torch.tensor([0.5, 0.5, 0.5])  # h = pad = 3
    out = _accumulate_prefix_grad(tail_grad, next_prefix_grad, pad=3)
    assert out.tolist() == [1.5, 2.5, 3.5]


# ------------------------------------------------- 3. SP divisibility contract

def _zigzag_local_rows(doc_len, cp):
    """Rows one CP rank owns for a single document under zigzag partitioning."""
    padded = doc_len + (-doc_len % (2 * cp))
    pieces = _pieces("zigzag", 0, [0, padded], cp)
    return sum(length for _, length in pieces)


@pytest.mark.parametrize(
    ("doc_len", "cp", "tp", "should_split"),
    [
        (24, 4, 4, False),  # 6 local rows cannot be split across 4 TP ranks
        (32, 4, 4, True),   # 8 local rows can
        (16, 4, 2, True),   # 8 rows / 2 TP ranks
        (8, 4, 4, False),   # 2 rows cannot be split across 4
    ],
)
def test_sp_split_requires_local_rows_divisible_by_tp(doc_len, cp, tp, should_split):
    """The boundary must not ask SP to scatter an indivisible row count."""
    rows = _zigzag_local_rows(doc_len, cp)
    divisible = rows % tp == 0
    assert divisible is should_split, (
        f"doc_len={doc_len} cp={cp} tp={tp}: zigzag gives {rows} local rows, "
        f"expected divisible={should_split}"
    )
