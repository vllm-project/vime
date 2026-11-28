"""CPU tests for the contiguous-CP layout helpers.

These pin the coordinate math that decides which global tokens each CP rank
owns, and how a token moves between the contiguous and zigzag conventions.
No NPU, no process group, no Megatron: the helpers are pure functions of
`(layout, rank, cu_seqlens, cp)`.

Coverage:
  * document boundaries that fall inside a rank's chunk (cross-rank docs),
  * lengths that require `2*cp` padding before zigzag partitioning,
  * padding that only widens the widest document (the RoPE table case),
  * token conservation through the contiguous -> zigzag redistribution plan.
"""

from __future__ import annotations

import pytest

from vime_plugins.models.gdn.cp_layout import _pieces, _split, _to_local


def _zigzag_padded(cu_seqlens, cp):
    """Pad every document up to a multiple of 2*cp, as the attention boundary does."""
    out = [0]
    for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        length = b - a
        out.append(out[-1] + length + (-length % (2 * cp)))
    return out


def _covered(rank, cu_seqlens, cp):
    """Global offsets owned by `rank` in the contiguous layout."""
    (start, length) = _pieces("contiguous", rank, cu_seqlens, cp)[0]
    return set(range(start, start + length))


@pytest.mark.parametrize("cp", [1, 2, 4])
def test_contiguous_partitions_cover_every_token_once(cp):
    cu = [0, 64, 192, 256]  # total 256, divisible by 1/2/4
    seen = []
    for rank in range(cp):
        seen.extend(sorted(_covered(rank, cu, cp)))
    assert seen == list(range(256))


def test_cross_rank_document_boundary():
    """A document that straddles a rank boundary must be split, not duplicated."""
    cp, cu = 4, [0, 96, 256]  # rank chunk = 64; doc0 spans 0..96, crossing rank 0 -> 1
    rank0, rank1 = _covered(0, cu, cp), _covered(1, cu, cp)
    assert rank0 == set(range(0, 64))
    assert rank1 == set(range(64, 128))
    # the split document contributes its first 64 tokens to rank 0 and the
    # remaining 32 to rank 1 -- counted once, on each side
    assert len(rank0 & set(range(0, 96))) == 64
    assert len(rank1 & set(range(0, 96))) == 32
    assert not (rank0 & rank1)


def test_zigzag_requires_2cp_alignment():
    cp = 4
    cu = [0, 100]  # 100 is not a multiple of 2*cp = 8
    with pytest.raises(RuntimeError, match="divisible by 2\\*cp"):
        _pieces("zigzag", 0, cu, cp)


def test_zigzag_after_padding_partitions_exactly():
    """Every rank owns exactly 1/cp of the padded stream, in mirrored halves."""
    cp = 4
    # 767 needs the RoPE pad; 257 (1024-767) is misaligned too, so it rounds to 264
    cu = [0, 767, 1024]
    padded = _zigzag_padded(cu, cp)
    assert padded == [0, 768, 1032]

    owned = []
    for rank in range(cp):
        pieces = _pieces("zigzag", rank, padded, cp)
        # two chunks per document: head slice and mirrored tail slice
        assert len(pieces) == 2 * (len(padded) - 1)
        owned += [t for start, length in pieces for t in range(start, start + length)]

    # each global offset of the padded stream appears exactly once
    assert sorted(owned) == list(range(padded[-1]))


def test_mirrored_halves_are_complementary():
    """Rank r's tail chunk mirrors rank (2*cp-1-r)'s head chunk."""
    cp = 4
    padded = _zigzag_padded([0, 96, 256], cp)
    rank0 = _pieces("zigzag", 0, padded, cp)
    rank7 = _pieces("zigzag", 7, padded, cp)
    # for a document of width w, rank r owns [r*ch, (r+1)*ch) and
    # [w-(r+1)*ch, w-r*ch); rank 7's head must sit right after rank 0's head
    doc = 96
    ch = doc // (2 * cp)
    r0_head = [p for p in rank0 if p[0] < doc][0]
    r7_head = [p for p in rank7 if p[0] < doc][0]
    assert r0_head == (0, ch)
    assert r7_head == (7 * ch, ch)


def test_split_never_duplicates_or_drops_tokens():
    cp = 4
    padded = _zigzag_padded([0, 96, 256], cp)
    per_rank = [_pieces("zigzag", r, padded, cp) for r in range(cp)]

    for src in range(cp):
        moved = _split(per_rank[src], per_rank)
        assert len(moved) == cp
        # `_split` intersects the source ranges with each peer's ranges, so the
        # total handed out equals what the source owned
        emitted = sum(length for dst_pieces in moved for _, length in dst_pieces)
        assert emitted == sum(length for _, length in per_rank[src])
        # and nothing is duplicated across destinations
        flattened = [t for dst_pieces in moved for start, length in dst_pieces
                     for t in range(start, start + length)]
        assert len(flattened) == len(set(flattened))


def test_self_split_is_identity():
    """A rank intersecting with its own ranges hands back exactly what it owned."""
    cp = 4
    padded = _zigzag_padded([0, 96, 256], cp)
    per_rank = [_pieces("zigzag", r, padded, cp) for r in range(cp)]
    for src in range(cp):
        mine = _split(per_rank[src], per_rank)[src]
        assert sorted(mine) == sorted(per_rank[src])


def test_to_local_maps_offsets_into_gathered_buffer():
    ranges = [(0, 64), (128, 32)]
    assert _to_local(0, ranges) == 0
    assert _to_local(63, ranges) == 63
    assert _to_local(128, ranges) == 64  # continues after the first piece
    assert _to_local(159, ranges) == 95
    with pytest.raises(KeyError):
        _to_local(64, ranges)  # gap between pieces is not owned


def test_padding_only_extends_the_widest_document():
    """The RoPE pad target is the widest padded doc, not the packed total.

    `cu` here is the observed contiguous-CP packing: the widest document is
    767 tokens, which the 2*cp alignment pushes to 768 -- one row of RoPE table
    beyond what `max_seqlen_q` reported when the table was built.
    """
    cp = 4
    cu = [0, 296, 754, 1059, 1281, 2048]
    padded = _zigzag_padded(cu, cp)
    widths_raw = [b - a for a, b in zip(cu[:-1], cu[1:])]
    widths_pad = [b - a for a, b in zip(padded[:-1], padded[1:])]

    assert max(widths_raw) == 767
    assert max(widths_pad) == 768
    # padding never shrinks a document, and only rounds up to 2*cp
    assert all(p >= r for p, r in zip(widths_pad, widths_raw))
    assert all(p % (2 * cp) == 0 for p in widths_pad)
