"""Context-parallel layout conversion for hybrid (GDN + GQA) stacks.

Two CP layout conventions coexist in a Qwen3.5-style model:

* **contiguous** (chunkwise linear attention / fla): the *flat* packed stream is
  split into ``cp`` equal intervals; rank ``r`` owns ``[r*T/cp, (r+1)*T/cp)``.
* **zigzag / per-document** (regular attention, TE ring CP): every sequence is
  split into ``2*cp`` chunks and rank ``r`` owns chunks ``[r, 2*cp-1-r]``.

Keeping the stack in one layout forces a fake CP for the linear-attention
layers.  Following the Megatron design (NVIDIA/Megatron-LM#7128) we keep the
stack **contiguous** and convert only at the attention boundary, as an
all-to-all redistribution (no token duplication: memory and compute stay 1x).
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _pieces(layout: str, rank: int, cu_seqlens: list[int], cp: int) -> list[tuple[int, int]]:
    """``[(global_start, length), ...]`` owned by ``rank`` in ``layout``."""
    if layout == "contiguous":
        total = cu_seqlens[-1]
        if total % cp:
            raise RuntimeError(f"contiguous CP needs total {total} divisible by cp {cp}")
        step = total // cp
        return [(rank * step, step)]
    if layout == "zigzag":
        out = []
        for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            length = b - a
            if length % (2 * cp):
                raise RuntimeError(
                    f"zigzag CP needs every sequence length ({length}) divisible by 2*cp ({2*cp})"
                )
            ch = length // (2 * cp)
            out.append((a + rank * ch, ch))
            out.append((a + (2 * cp - 1 - rank) * ch, ch))
        return out
    raise ValueError(f"unknown cp layout {layout!r}")


def _split(ranges, per_rank):
    """Intersect local ranges with the ranges owned by each peer."""
    out: list[list[tuple[int, int, int]]] = [[] for _ in per_rank]
    for start, length in ranges:
        end = start + length
        for dst, dst_pieces in enumerate(per_rank):
            for ds, dl in dst_pieces:
                lo, hi = max(start, ds), min(end, ds + dl)
                if lo < hi:
                    out[dst].append((lo, hi - lo))
    return out



def _to_local(offset: int, ranges: list[tuple[int, int]]) -> int:
    """Map a global token offset into the local buffer built from ``ranges``."""
    local = 0
    for start, length in ranges:
        if start <= offset < start + length:
            return local + (offset - start)
        local += length
    raise KeyError(f"offset {offset} not covered by {ranges}")


def make_plan(
    src_layout: str, dst_layout: str, rank: int, cp: int, cu_seqlens: list[int], padded_cu=None
):
    """Pure routing plan for one rank.

    Returns ``(src_ranges, dst_ranges, send, recv)`` where ``send[d]`` lists the
    global ``(start, length)`` pieces this rank ships to rank ``d`` and
    ``recv[s]`` lists the pieces it receives from rank ``s``.
    """
    def pieces(layout, peer):
        if padded_cu is None:
            return _pieces(layout, peer, cu_seqlens, cp)
        if layout == "zigzag":
            return _pieces(layout, peer, padded_cu, cp)

        # Express the real contiguous shard in the padded global coordinate
        # system. Its local buffer still contains only the original tokens.
        (start, length), = _pieces(layout, peer, cu_seqlens, cp)
        end = start + length
        ranges = []
        for lo, hi, padded_start in zip(cu_seqlens[:-1], cu_seqlens[1:], padded_cu[:-1], strict=True):
            left, right = max(lo, start), min(hi, end)
            if left < right:
                ranges.append((padded_start + left - lo, right - left))
        return ranges

    sources = [pieces(src_layout, r) for r in range(cp)]
    targets = [pieces(dst_layout, r) for r in range(cp)]
    src, dst = sources[rank], targets[rank]
    send = _split(src, targets)
    recv = _split(dst, sources)
    return src, dst, send, recv


def _redistribute(
    x: torch.Tensor,
    src_layout: str,
    dst_layout: str,
    cu_seqlens: list[int],
    group: dist.ProcessGroup | None,
    padded_cu=None,
) -> torch.Tensor:
    """Raw collective: redistribute ``x`` (``[1, T_local, ...]``) between layouts."""
    if src_layout == dst_layout:
        return x
    cp = dist.get_world_size(group) if group is not None else 1
    if cp == 1:
        return x
    # all_to_all_single splits along dim 0, while packed THD tensors carry the
    # sequence on dim 1 ([1, T, ...]).  Squeeze the singleton batch dim so the
    # collective operates on tokens, then restore it on the way out.
    squeeze = x.dim() > 1 and x.shape[0] == 1
    xt = x.squeeze(0) if squeeze else x
    rank = dist.get_rank(group)
    cu = list(cu_seqlens)
    src_ranges, dst_ranges, send, recv = make_plan(src_layout, dst_layout, rank, cp, cu, padded_cu)
    if sum(l for _, l in src_ranges) != xt.shape[0]:
        raise RuntimeError(
            f"local length mismatch: layout says {sum(l for _, l in src_ranges)}, tensor has {xt.shape[0]}"
        )

    trailing = tuple(xt.shape[1:])
    # ---- send: split my source ranges into pieces for each destination ----
    # Rebuild properly: map global offsets to local offsets of my source buffer.
    input_sizes, send_parts = [], []
    for dst in range(cp):
        size = 0
        for start, length in send[dst]:
            lo = _to_local(start, src_ranges)
            send_parts.append(xt[lo : lo + length])
            size += length
        input_sizes.append(size)

    # ---- receive: which source pieces land in my destination ranges ----
    output_sizes = [sum(l for _, l in recv[p]) for p in range(cp)]

    inp = torch.cat(send_parts, dim=0) if send_parts else xt.new_empty((0, *trailing))
    out = xt.new_empty((sum(output_sizes), *trailing))
    dist.all_to_all_single(
        out,
        inp.contiguous(),
        output_split_sizes=output_sizes,
        input_split_sizes=input_sizes,
        group=group,
    )

    # ---- assemble in destination order ----
    total = sum(length for _, length in dst_ranges)
    if padded_cu is not None and dst_layout == "zigzag":
        result = xt.new_zeros((total, *trailing))
    else:
        result = xt.new_empty((total, *trailing))
    pos = 0
    for peer in range(cp):
        for start, length in recv[peer]:
            lo = _to_local(start, dst_ranges)
            result[lo : lo + length] = out[pos : pos + length]
            pos += length
    return (result.unsqueeze(0) if squeeze else result).contiguous()


class _LayoutConvert(torch.autograd.Function):
    """Autograd wrapper for the layout redistribution.

    Backward reverses the redistribution. Padding is zero-filled on entry
    to zigzag and discarded on return to contiguous.
    """

    @staticmethod
    def forward(ctx, x, src_layout, dst_layout, cu_seqlens, group, padded_cu=None):
        ctx.args = (dst_layout, src_layout, cu_seqlens, group, padded_cu)
        return _redistribute(x, src_layout, dst_layout, cu_seqlens, group, padded_cu)

    @staticmethod
    def backward(ctx, grad_output):
        dst_layout, src_layout, cu_seqlens, group, padded_cu = ctx.args
        grad = _redistribute(grad_output, dst_layout, src_layout, cu_seqlens, group, padded_cu)
        return grad, None, None, None, None, None


def convert_layout(x, src_layout, dst_layout, cu_seqlens, group, padded_cu=None):
    """Differentiable layout conversion between CP conventions."""
    if src_layout == dst_layout:
        return x
    if _skip_convert():
        return x
    return _LayoutConvert.apply(x, src_layout, dst_layout, cu_seqlens, group, padded_cu)


def _skip_convert():
    import os

    return os.environ.get("VIME_CP_SKIP_CONVERT") == "1"


def contiguous_to_zigzag(x, cu_seqlens, group=None, *, padded_cu=None):
    return convert_layout(x, "contiguous", "zigzag", cu_seqlens, group, padded_cu)


def zigzag_to_contiguous(x, cu_seqlens, group=None, *, padded_cu=None):
    return convert_layout(x, "zigzag", "contiguous", cu_seqlens, group, padded_cu)
