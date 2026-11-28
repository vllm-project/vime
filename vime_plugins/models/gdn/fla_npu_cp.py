"""AscendC (``fla_npu``) kernels for FLA's context-parallel affine chain.

FLA's CP implementation (``fla/ops/cp``) composes the cross-rank state with an
affine chain: every rank summarises its segment as ``[h | m]`` — the zero-input
response ``h`` (K×V) and the transition matrix ``m`` (K×K) — the summaries are
all-gathered, and each rank folds its predecessors via ``h ← m_i @ h + h_i``.

FLA runs that with Triton kernels.  ``fla_npu`` 26.10.0 ships AscendC
counterparts with matching semantics:

    npu_pre_process_fwd_kernel_merged(k, w, u, g=…, cu_seqlens=…, chunk_size=…)
        -> hm[Nseq, HV, K, V+K] fp32; left V columns = h, right K columns = m
    npu_merge_fwd_bwd_kernel(h, ag_hm, num_ranks, rank, forward=…, state_v_first=…)
        -> h written in place; `h ← He_0; h ← M_i @ h + He_i`
    npu_pre_process_bwd_kernel_merged(q, k, w, d_o, dv, scale, chunk_size, …)
        -> dhm[HV, K, V+K] fp32; left V = E_r, right K = P_r

``npu_merge_fwd_bwd_kernel`` documents "argument order matches FLA", so the
merge is a drop-in.  ``pre_process`` returns a fresh tensor while FLA's kernel
writes into a caller-provided ``hm``, hence the copy in the wrapper.

Usage
-----
    from vime_plugins.models.gdn.fla_npu_cp import install
    install()          # before the first CP forward

``install()`` patches the kernel names inside ``fla.ops.cp.chunk_delta_h`` so
the whole CP orchestration (layout handling, all-gather, chain order) is reused
unchanged while the heavy kernels run on AscendC.
"""

from __future__ import annotations

import warnings

import torch

from fla_npu.ops.ascendc._aclnn_ctypes import (
    npu_merge_fwd_bwd_kernel,
    npu_pre_process_bwd_kernel_merged,
    npu_pre_process_fwd_kernel_merged,
)


def chunk_cp_fwd_h(
    k,
    w,
    u,
    hm,
    g=None,
    gk=None,
    bg=None,
    v=None,
    cu_seqlens=None,
    chunk_size=64,
    **_ignored,
):
    """AscendC replacement for FLA's ``chunk_cp_fwd_h`` (writes ``hm`` in place).

    FLA hands over ``[B, T, H, D]``; the AscendC kernel wants ``[B, H, T, D]``.
    """
    out = npu_pre_process_fwd_kernel_merged(
        k.transpose(1, 2).contiguous(),
        w.transpose(1, 2).contiguous(),
        u.transpose(1, 2).contiguous(),
        g=None if g is None else g.transpose(1, 2).contiguous(),
        gk=None if gk is None else gk.transpose(1, 2).contiguous(),
        bg=bg,
        v=v,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    if tuple(out.shape) != tuple(hm.shape):
        out = out.reshape(hm.shape)
    hm.copy_(out.to(hm.dtype))
    return hm


def chunk_cp_merge(
    h,
    ag_hm,
    num_ranks,
    rank,
    state_v_first=False,
    forward=True,
    h_seq_idx=None,
    use_graph=False,
    **_ignored,
):
    """AscendC replacement for FLA's ``merge_fwd_bwd_kernel`` (in-place fold)."""
    del h_seq_idx, use_graph
    npu_merge_fwd_bwd_kernel(
        h,
        ag_hm,
        num_ranks,
        rank,
        forward=forward,
        state_v_first=state_v_first,
    )
    return h


def chunk_cp_bwd_dhu_pre_process(
    q,
    k,
    w,
    d_o,
    dv,
    scale,
    chunk_size,
    g=None,
    gk=None,
    cu_seqlens=None,
    **_ignored,
):
    """AscendC replacement for the CP backward state pre-process."""
    return npu_pre_process_bwd_kernel_merged(
        q,
        k,
        w,
        d_o,
        dv,
        float(scale),
        int(chunk_size),
        g=g,
        gk=gk,
        cu_seqlens=cu_seqlens,
    )


_PATCHED = False


def install(verbose: bool = True) -> bool:
    """Patch ``fla.ops.cp`` to use the AscendC kernels. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import fla.ops.cp.chunk_delta_h as cp_delta_h
    except ImportError as exc:  # pragma: no cover - fla present in this env
        warnings.warn(f"fla_npu_cp: cannot import fla.ops.cp.chunk_delta_h: {exc}")
        return False

    patched = []
    for name, impl in (
        ("chunk_cp_fwd_h", chunk_cp_fwd_h),
        ("chunk_cp_merge", chunk_cp_merge),
        ("chunk_cp_bwd_dhu_pre_process", chunk_cp_bwd_dhu_pre_process),
    ):
        if hasattr(cp_delta_h, name):
            setattr(cp_delta_h, name, impl)
            patched.append(name)

    _PATCHED = True
    if verbose:
        print(f"[fla_npu_cp] patched fla.ops.cp.chunk_delta_h: {patched}")
        missing = [
            n
            for n in ("chunk_cp_fwd_h", "chunk_cp_merge", "chunk_cp_bwd_dhu_pre_process")
            if n not in patched
        ]
        if missing:
            print(f"[fla_npu_cp] NOT FOUND (check the fla revision): {missing}")
    return True
