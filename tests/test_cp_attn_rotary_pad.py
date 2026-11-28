"""Regression tests for the RoPE table padding at the contiguous-CP boundary.

The attention boundary pads every packed document up to `2*cp` alignment.
That can widen the widest document by one row (observed 767 -> 768), while the
RoPE table was built earlier from the *unpadded* maximum. `CPAttnBoundary`
therefore extends the table to the widest padded document before handing it to
the attention kernel.

These tests call the production helper directly, so a regression in the pad
target (for example comparing against the rank's total packed rows, 516,
instead of the widest document, 768) fails here.
"""

from __future__ import annotations

import pytest
import torch

from vime_plugins.models.cp_attn_boundary import (
    pad_rotary_to_widest_document,
    widest_padded_document,
)


CP = 4
# Observed contiguous-CP packing whose widest document is 767 tokens.
CU = [0, 296, 754, 1059, 1281, 2048]
# The attention boundary aligns every document to 2*cp = 8.
PADDED_CU = [0, 296, 760, 1072, 1296, 2064]


def test_widest_padded_document_is_768():
    assert widest_padded_document(PADDED_CU) == 768
    # and the shape that triggered the original failure
    assert max(b - a for a, b in zip(CU[:-1], CU[1:])) == 767


def test_table_is_padded_to_the_widest_document():
    """767-row table + widest padded doc 768 -> one appended row."""
    rotary = torch.arange(767, dtype=torch.float32).unsqueeze(-1).repeat(1, 4)
    padded = pad_rotary_to_widest_document(rotary, PADDED_CU)

    assert padded.shape[0] == 768
    assert padded.shape[1:] == rotary.shape[1:]
    # every real position keeps its own row
    assert torch.equal(padded[:767], rotary)
    # the appended row repeats the last one
    assert torch.equal(padded[767], rotary[766])


def test_the_buggy_target_would_not_pad():
    """Guard the regression: a 516-row target must not be what we compare to.

    This is the exact failure mode the old code had -- `total packed rows`
    (516) is smaller than the table (767), so `extra < 0` and nothing was
    padded even though 768 rows were needed.
    """
    local_rows = sum(b - a for a, b in zip(PADDED_CU[:-1], PADDED_CU[1:])) // CP
    assert local_rows == 516
    assert local_rows < 767  # comparing against this would skip the pad
    # the production helper compares against the widest document instead
    assert widest_padded_document(PADDED_CU) > 767


def test_already_long_enough_table_is_returned_unchanged():
    rotary = torch.zeros(1024, 4)
    out = pad_rotary_to_widest_document(rotary, PADDED_CU)
    assert out is rotary


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
def test_alignment_is_idempotent(cp):
    """A document already aligned to 2*cp needs no further widening."""
    width = 2 * cp * 3
    cu = [0, width, 2 * width]
    assert widest_padded_document(cu) == width
    rotary = torch.zeros(width, 2)
    assert pad_rotary_to_widest_document(rotary, cu) is rotary
