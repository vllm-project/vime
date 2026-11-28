"""AscendC context-parallel GDN (affine chain), forward + backward.

The forward mirrors FLA's ``chunk_gated_delta_rule_fwd`` with AscendC kernels:

    prepare -> w, u, A, g_cumsum        (npu_chunk_gated_delta_rule_fwd_prepare)
    summary -> hm = [h | m] per segment (npu_pre_process_fwd_kernel_merged)
    merge   -> this rank's h0           (npu_merge_fwd_bwd_kernel)
    fwd_h   -> per-chunk states         (npu_chunk_gated_delta_rule_fwd_h)
    fwd_o   -> output                   (npu_chunk_fwd_o, BNSD only on A3)

The backward inserts the CP state-gradient: ``dht`` is produced by
``chunk_gated_delta_rule_bwd_dhu_pre_process`` (FLA orchestration, AscendC
kernels via ``fla_npu_cp_backend``) once the local ``dv`` exists, which is
exactly where ``flash_chunk_gated_delta_rule_bwd``'s ``cp_dht_provider`` hook
fires.

Layouts: q, k [B, H, T, K]; v [B, H, T, V]; g, beta [B, T, H] on the way in;
``o`` comes back as [B, T, H, V].
"""

from __future__ import annotations

import torch


def _npu_ops():
    from fla_npu.ops.ascendc._aclnn_ctypes import (
        _chunk_indices_from_cu_seqlens as chunk_idx,
        npu_chunk_fwd_o as fwd_o,
        npu_chunk_gated_delta_rule_fwd_h as fwd_h,
        npu_chunk_gated_delta_rule_fwd_prepare as prepare,
    )

    return prepare, fwd_h, fwd_o, chunk_idx


class CPGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, cp_context, chunk_size):
        from vime_plugins.models.gdn.fla_npu_cp_backend import install as install_ascendc_cp

        install_ascendc_cp(verbose=True)

        from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process

        prepare, fwd_h, fwd_o, chunk_idx = _npu_ops()

        t_local = int(q.shape[2])
        # Use the rank-local, per-document boundaries from the CP context; a
        # contiguous shard is usually several packed documents, not one.
        ctx_cu = getattr(cp_context, "cu_seqlens", None) if cp_context is not None else None
        if ctx_cu is None:
            cu = [0, t_local]
        else:
            cu = [int(v) for v in (ctx_cu.tolist() if hasattr(ctx_cu, "tolist") else ctx_cu)]
            if cu[-1] != t_local:
                import torch.distributed as _d
                import os as _os

                _r = _d.get_rank(cp_context.group) if cp_context.group is not None else 0
                raise RuntimeError(
                    f"[gdn-cp] cu_seqlens mismatch on rank {_r}: cu[-1]={cu[-1]} "
                    f"tokens={t_local} cu={cu}; refusing to drop document boundaries"
                )
        ci = chunk_idx(cu, chunk_size)

        # fla_npu's public contract is g/beta = [B, T, H]; the AscendC kernels
        # want [B, HV, T].  Convert only for the kernel calls.
        g_bh = g.transpose(1, 2).contiguous() if g.dim() == 3 else g
        beta_bh = beta.transpose(1, 2).contiguous() if beta.dim() == 3 else beta

        # Same as ChunkGatedDeltaRuleFunction: normalise in Python, then use the
        # normalised tensors for the whole chain (prepare / fwd_h / fwd_o / bwd).
        from fla_npu.ops.triton import l2norm_fwd as _l2norm_fwd

        q_n, q_rstd = _l2norm_fwd(q)
        k_n, k_rstd = _l2norm_fwd(k)
        q_n = q_n.to(q.dtype)
        k_n = k_n.to(k.dtype)

        _qh, _kh, _qr2, _kr2, beta_out, g_cumsum, w, u, A = prepare(
            q_n, k_n, v, g_bh, beta_bh,
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu,
        )

        h0 = chunk_gated_delta_rule_fwd_h_pre_process(
            # cross-rank state must be built from the SAME normalised k the
            # local recurrence uses, otherwise the merged h0 is inconsistent
            # with the chunks that consume it
            k=k_n.transpose(1, 2).contiguous(),
            w=w.transpose(1, 2).contiguous(),
            u=u.transpose(1, 2).contiguous(),
            g=g_cumsum.transpose(1, 2).contiguous(),
            cu_seqlens=cu,
            initial_state=None,
            context=cp_context,
            state_v_first=False,
            chunk_size=chunk_size,
        )
        if h0 is not None:
            want_tail = (k.shape[-1], v.shape[-1])
            if tuple(h0.shape[-2:]) != want_tail:
                h0 = h0.reshape(h0.shape[0], h0.shape[1], *want_tail)
            # Keep fp32: the fused path also accumulates its state in fp32,
            # and rounding h0 to bf16 makes every downstream token inherit the
            # rounding error.
            if h0.dtype not in (torch.float32, k.dtype):
                h0 = h0.float()


        h, v_new, final_state = fwd_h(
            k_n, w, u,
            g=g_cumsum,
            gk=None,
            initial_state=h0,
            output_final_state=True,
            chunk_size=chunk_size,
            cu_seqlens=cu,
            chunk_indices=ci,
        )

        o = fwd_o(
            q_n, k_n, v_new, h,
            float(scale),
            g=g_cumsum,
            cu_seqlens=cu,
            chunk_indices=ci,
            chunk_size=chunk_size,
        )
        # A3 only supports BNSD output from ChunkFwdO (BSND/TND need A5).
        o = o.transpose(1, 2).contiguous()

        # The backward expects the *cumsum'd* gate (as fla_npu's own
        # autograd Function saves), in [B, T, H] layout.
        # w/u and the row-major g_cumsum are NOT read by backward (verified):
        # `flash_chunk_gated_delta_rule_bwd` recomputes w internally, and only
        # the transposed g is consumed. Keeping them here only extends their
        # lifetime by ~0.57 GiB/card at 8K/CP=2 (0.8B).
        ctx.save_for_backward(
            q_n, k_n, v, g_cumsum.transpose(1, 2).contiguous(), beta, A, h0
        )
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.scale = float(scale)
        ctx.cp_context = cp_context
        ctx.chunk_size = int(chunk_size)
        ctx.cu = cu
        ctx.chunk_idx = ci
        return o

    @staticmethod
    def backward(ctx, do):
        from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu_pre_process

        from vime_plugins.models.gdn.fla_npu import flash_chunk_gated_delta_rule_bwd

        q, k, v, g, beta, A, h0 = ctx.saved_tensors
        cu = ctx.cu
        cp_context = ctx.cp_context
        chunk_size = ctx.chunk_size

        do_bh = do  # the backward helper transposes do internally

        # fla_npu expects per-chunk-size chunk-index metadata (a dict), because
        # recompute_w_u / bwd_dv_local / bwd_dhu may use different block sizes.
        from fla_npu.ops.ascendc._aclnn_ctypes import (
            _chunk_indices_from_cu_seqlens as _ci,
        )

        ci_dict = {str(s): _ci(cu, s) for s in (32, 64, 128, chunk_size)}

        def dht_provider(*, q, k, w, do, dv, g, scale, chunk_size):
            res = chunk_gated_delta_rule_bwd_dhu_pre_process(
                q=q.transpose(1, 2).contiguous(),
                k=k.transpose(1, 2).contiguous(),
                w=w.transpose(1, 2).contiguous(),
                do=do.transpose(1, 2).contiguous(),
                dv=dv.transpose(1, 2).contiguous(),
                g=g.transpose(1, 2).contiguous(),
                scale=scale,
                cu_seqlens=cu,
                context=cp_context,
                chunk_size=chunk_size,
            )
            dht = res[0] if isinstance(res, (tuple, list)) else res
            if dht is not None and dht.dtype != q.dtype:
                dht = dht.to(q.dtype)
            try:
                import torch.distributed as _d

                _r = _d.get_rank(cp_context.group)
                _m = 0.0 if dht is None else float(dht.abs().max())
                print(f"[cp] rank={_r} dht is None={dht is None} absmax={_m:.3e}", flush=True)
            except Exception as _e:  # pragma: no cover
                print(f"[cp] dht debug failed: {_e}", flush=True)
            return dht

        dq, dk, dv, db, dg, _dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            do=do_bh,
            cu_seqlens=None,
            cu_seqlens_list=cu,
            chunk_indices=None,
            chunk_indices_list=ci_dict,
            chunk_size=chunk_size,
            initial_state=h0,
            cp_dht_provider=dht_provider,
        )
        from fla_npu.ops.triton import l2norm_bwd as _l2norm_bwd

        if getattr(ctx, "q_rstd", None) is not None:
            dq = _l2norm_bwd(ctx.saved_tensors[0], ctx.q_rstd, dq)
        if getattr(ctx, "k_rstd", None) is not None:
            dk = _l2norm_bwd(ctx.saved_tensors[1], ctx.k_rstd, dk)
        return dq, dk, dv, dg, db, None, None, None


def flash_gated_delta_rule_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    cp_context=None,
    chunk_size: int = 64,
):
    """CP forward (differentiable) for one GDN layer on the local shard."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return CPGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, float(scale), cp_context, int(chunk_size)
    ), None
