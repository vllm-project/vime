"""
fla-npu backed GDN kernels for Qwen3.5/3.6 on Ascend NPU.

flash-linear-attention (``fla``) ships ``FusedRMSNormGated``,
``ShortConvolution`` and ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``,
none of which exist in the ``fla_npu`` package. ``fla_npu`` only exposes
low-level Ascend C operator primitives via ``fla_npu.ops.ascendc``.

The high-level, autograd-wrapped GDN entry point and the causal-conv1d /
RMSNorm-gated modules were ported from
``flash-linear-attention-npu/examples/flash_gated_delta_rule.py`` so that the
Qwen3.5/3.6 linear-attention layer can run on NPU without the ``fla`` package.

This module intentionally supports only what the Qwen3.5/3.6 GDN training
path needs:

* No recurrent-state support: ``flash_gated_delta_rule`` never takes an
  ``initial_state`` and always returns ``(o, None)``; the causal conv never
  takes an ``initial_state`` / ``residual`` and never outputs a final state.
* The causal conv applies SiLU inside the AscendC kernel (matching the
  vllm-ascend rollout path) and is bias-free, like the HF checkpoint.

Layout contract (matches the upstream example):
    q, k: [B, H, T, K]
    v:    [B, H, T, V]
    g:    [B, T, H]
    beta: [B, T, H]
    returns o: [B, T, H, V]
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_npu  # noqa: F401  # registers the NPU backend; used by QwenStyleRMSNormGated

from fla_npu.ops import ascendc as ascendc_ops
from fla_npu.ops.ascendc import causal_conv1d_bwd as ascendc_causal_conv1d_bwd
from fla_npu.ops.ascendc import causal_conv1d_fn as ascendc_causal_conv1d_fn
from fla_npu.ops.ascendc import chunk_bwd_dqkwg as ascendc_chunk_bwd_dqkwg
from fla_npu.ops.ascendc import chunk_bwd_dv_local as ascendc_chunk_bwd_dv_local
from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu as ascendc_chunk_gated_delta_rule_bwd_dhu
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd as ascendc_chunk_gated_delta_rule_fwd
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h as ascendc_chunk_gated_delta_rule_fwd_h
from fla_npu.ops.ascendc import prepare_wy_repr_bwd_da as ascendc_prepare_wy_repr_bwd_da
from fla_npu.ops.ascendc import prepare_wy_repr_bwd_full as ascendc_prepare_wy_repr_bwd_full
from fla_npu.ops.ascendc import recompute_w_u_fwd as ascendc_recompute_w_u_fwd
from fla_npu.ops.triton import autocast_custom_bwd, autocast_custom_fwd, input_guard, l2norm_bwd, l2norm_fwd

_disable_compile = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)
_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def cdiv_torch(a, b):
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_list(cu_seqlens, chunk_size: int):
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = [int(x) for x in cu_seqlens.detach().cpu().tolist()]

    indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _as_int_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _flatten_varlen_x(x: torch.Tensor, cu_seqlens):
    if cu_seqlens is None:
        return x, None, False
    cu_list = _as_int_list(cu_seqlens)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError("causal_conv1d varlen path expects x.shape[0] == 1 for [1, T, D] input.")
        return x.reshape(x.shape[1], x.shape[2]).contiguous(), cu_list, True
    if x.ndim == 2:
        return x.contiguous(), cu_list, True
    raise ValueError(f"causal_conv1d varlen path expects rank-2 or rank-3 input, got shape={tuple(x.shape)}.")


# ---------------------------------------------------------------------------
# varlen metadata (cu_seqlens / chunk indices for the AscendC kernels)
# ---------------------------------------------------------------------------


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _cumsum_block_t(g: torch.Tensor, chunk_size: int) -> int:
    if g.dim() != 3:
        raise ValueError(f"AscendC chunk_local_cumsum expects rank-3 input, got shape={tuple(g.shape)}.")
    return _next_power_of_two((1 << 17) // int(chunk_size))


def _ensure_varlen_metadata(g, cu_seqlens, chunk_size):
    """Build the cu_seqlens / chunk-index metadata for the whole fwd/bwd chain.

    The AscendC kernels take Python-list cu_seqlens / chunk_indices while some
    helpers take tensors, so both representations are precomputed for every
    chunk size any kernel in the chain may request.
    """
    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_seqlens_list = _as_int_list(cu_seqlens)
    cumsum_block_t = _cumsum_block_t(g, chunk_size)
    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(int(chunk_size))
    required_sizes.add(cumsum_block_t)
    tensor_indices = {}
    list_indices = {}
    for size in required_sizes:
        key = str(size)
        tensor_indices[key] = prepare_chunk_indices(cu_seqlens, size)
        list_indices[key] = prepare_chunk_indices_list(cu_seqlens_list, size)
    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def _chunk_tensor(chunk_indices, chunk_size: int):
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(chunk_indices_list, chunk_size: int):
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def recompute_w_u(k, v, beta, A, g, *, chunk_size, cu_seqlens, chunk_indices):
    w, u = ascendc_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u


def chunk_local_cumsum_bwd(dg, *, cu_seqlens, chunk_indices, chunk_size):
    """Reverse chunk-local cumsum of the g gradient ([B, T, H] in and out)."""
    # AscendC chunk_local_cumsum expects [B, H, T] head-first input and
    # float32 output; the chunk indices are looked up at the cumsum block
    # size, matching what `_ensure_varlen_metadata` precomputes.
    dg_bht = dg.transpose(1, 2).contiguous()
    cumsum_block_t = _cumsum_block_t(dg_bht, chunk_size)
    op_kwargs = {
        "reverse": True,
        "scale": 1.0,
        "head_first": True,
        "output_dtype": "float32",
    }
    cu_list = _as_int_list(cu_seqlens)
    chunk_list = _as_int_list(_chunk_tensor(chunk_indices, cumsum_block_t))
    if cu_list is not None or chunk_list is not None:
        op_kwargs["cu_seqlens"] = cu_list
        op_kwargs["chunk_indices_out"] = chunk_list
    out = ascendc_ops.chunk_local_cumsum(dg_bht.float(), chunk_size, **op_kwargs)
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Causal conv1d (autograd-wrapped)
# ---------------------------------------------------------------------------


class AscendCCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        cu_seqlens: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
    ):
        op_weight = weight.transpose(-1, -2).contiguous()
        width, dim = op_weight.shape
        op_x, query_start_loc, is_varlen = _flatten_varlen_x(x, cu_seqlens)
        num_sequences = len(query_start_loc) - 1 if query_start_loc is not None else int(x.shape[0])
        # ``conv_state`` carries the previous CP rank's last width-1 inputs;
        # without it the conv starts from a zero state (non-CP path).
        # ``conv_state`` is usually None or a plain tensor; autograd only accepts a
        # gradient for it when it was a leaf Variable, so remember which.
        ctx.state_wants_grad = bool(
            isinstance(conv_state, torch.Tensor) and conv_state.requires_grad
        )
        if conv_state is None:
            conv_states = torch.zeros(num_sequences, width - 1, dim, dtype=x.dtype, device=x.device)
        else:
            conv_states = (
                conv_state.to(dtype=x.dtype)
                .reshape(num_sequences, width - 1, dim)
                .contiguous()
            )
        query_start_loc_device = (
            cu_seqlens.to(device=op_x.device, dtype=torch.int32).contiguous() if cu_seqlens is not None else None
        )

        # The AscendC kernel applies SiLU in-kernel when activation != "none",
        # matching the rollout (vllm-ascend) path.  The in-kernel Silu() and
        # the Python F.silu (npu_silu) diverge under bfloat16, so the kernel
        # path is used for the forward output to keep training/rollout
        # logprobs aligned.
        y = ascendc_causal_conv1d_fn(
            op_x,
            op_weight,
            None,
            conv_states,
            query_start_loc=query_start_loc_device,
            has_initial_state=None,
            activation="silu",
            pad_slot_id=-1,
            null_block_id=None,
            head_num=H,
        )
        if is_varlen:
            y = y.unsqueeze(0)

        # The backward kernel needs the pre-activation conv output (silu
        # input) as its ``y`` argument; it cannot be recovered from the silu
        # output, so the preactivation is recomputed in the backward pass
        # (one extra conv call there) rather than saved here.
        ctx.save_for_backward(x, op_weight)
        ctx.conv_state = conv_state
        ctx.query_start_loc = query_start_loc
        ctx.is_varlen = is_varlen
        ctx.head_num = H
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, op_weight = ctx.saved_tensors
        op_x = x.reshape(-1, x.shape[-1]).contiguous() if ctx.is_varlen and x.ndim == 3 else x.contiguous()
        op_dy = dy.squeeze(0).contiguous() if ctx.is_varlen and dy.ndim == 4 else dy.contiguous()

        # Recompute the pre-activation conv output with activation=None so
        # the backward kernel gets the silu input it expects as its ``y``.
        # The bwd kernel recomputes the conv from x itself, so conv_states
        # is not restored here (passed as None).
        qsl_device = None
        if ctx.query_start_loc is not None:
            qsl_device = torch.tensor(ctx.query_start_loc, dtype=torch.int32, device=op_x.device).contiguous()
        preactivation = ascendc_causal_conv1d_fn(
            op_x,
            op_weight,
            None,
            None,
            query_start_loc=qsl_device,
            has_initial_state=None,
            activation=None,
            pad_slot_id=-1,
            null_block_id=None,
            head_num=ctx.head_num,
        )
        op_y = (
            preactivation.squeeze(0).contiguous()
            if ctx.is_varlen and preactivation.ndim == 4
            else preactivation.contiguous()
        )

        dx, dw, _db, dconv_state = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=op_y,
            weight=op_weight,
            dy=op_dy,
            initial_state=ctx.conv_state,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=1,  # silu
            input_layout="NTD" if ctx.is_varlen else "BNSD",
        )
        dx = dx.reshape_as(x)
        dw = dw.transpose(0, 1).contiguous()
        return dx, dw, None, None, (dconv_state if ctx.state_wants_grad else None)


def causal_conv1d_ascendc(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    cu_seqlens: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
) -> torch.Tensor:
    return AscendCCausalConv1dFunction.apply(x, weight, H, cu_seqlens, conv_state)


# ---------------------------------------------------------------------------
# GDN forward / backward
# ---------------------------------------------------------------------------


def flash_chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    cu_seqlens_list=None,
    chunk_indices_list=None,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = False,
):
    chunk_list = _chunk_list(chunk_indices_list, chunk_size)
    if torch.is_tensor(initial_state):
        initial_state = initial_state.to(q.dtype)
    # The composite AscendC kernel runs the whole forward (cumsum, kkt,
    # solve_tri, WY repr, recurrent, output) and returns the cumsum'd g and A
    # needed by the backward pass.
    import os as _os

    _init = initial_state
    if _init is None and _os.environ.get("VIME_FLA_NPU_INIT") == "1":
        # probe only: a fixed non-zero state, to see whether the op honours it
        _seed = int(_os.environ.get("VIME_FLA_NPU_INIT_SEED", "7"))
        _gen = torch.Generator(device=q.device).manual_seed(_seed)
        _dtype = q.dtype if _os.environ.get("VIME_FLA_NPU_INIT_DTYPE", "same") == "same" else torch.float32
        _order = _os.environ.get("VIME_FLA_NPU_INIT_ORDER", "kv")
        _k, _v = k.shape[-1], v.shape[-1]
        _shape = (q.shape[0], q.shape[1], _v, _k) if _order == "vk" else (q.shape[0], q.shape[1], _k, _v)
        _init = torch.randn(*_shape, device=q.device, dtype=_dtype, generator=_gen)
    o, final_state, g, A, *_ = ascendc_chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=_init,
        output_final_state=bool(output_final_state or _init is not None),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
        scale=scale,
        disable_recompute=True,
        return_intermediate_states=False,
        use_qk_l2norm_in_kernel=False,
    )
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q,
    k,
    v,
    g,
    beta,
    A,
    scale,
    do,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_indices=None,
    chunk_indices_list=None,
    chunk_size: int = 64,
    initial_state=None,
    dht=None,
    cp_dht_provider=None,
):
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    # AscendC requires the state / state-gradient dtype to match q.
    if torch.is_tensor(initial_state):
        initial_state = initial_state.to(q.dtype)
    if torch.is_tensor(dht):
        dht = dht.to(q.dtype)

    w, u = recompute_w_u(
        k,
        v,
        beta,
        A,
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )
    do = do.transpose(1, 2).contiguous()

    h, v_new, _ = ascendc_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    if __import__("os").environ.get("VIME_DBG_DVLOCAL") == "1":
        print("[dbg dv_local]",
              "q", tuple(q.shape), q.dtype,
              "k", tuple(k.shape),
              "do", tuple(do.shape),
              "g", None if g is None else tuple(g.shape),
              "A", None if A is None else tuple(A.shape),
              "scale", scale, "cs", chunk_size,
              "cu", cu_seqlens_list, "ci", _chunk_list(chunk_indices_list, chunk_size),
              flush=True)
    dv = ascendc_chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    # Context parallelism: the incoming state-gradient depends on the sibling
    # ranks' summaries, which need the local dv computed above.
    if cp_dht_provider is not None:
        dht = cp_dht_provider(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )

    dh, dh0, dv = ascendc_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=initial_state,
        dht=dht,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    dq, dk, dw, dg = ascendc_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dA = ascendc_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dk2, dv, db, dg2 = ascendc_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()
    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")

    db = db.transpose(1, 2).contiguous()
    dg = chunk_local_cumsum_bwd(
        dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens=None,
        cu_seqlens_list=None,
        chunk_indices=None,
        chunk_indices_list=None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        initial_state=None,
        output_final_state: bool = False,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        ctx.initial_state = initial_state
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dfinal_state=None):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            do=do,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
            initial_state=ctx.initial_state,
            dht=dfinal_state,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dh0,
            None,
        )


@_disable_compile
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    r"""
    Flash-linear-attention NPU port of xtuner's GDN entry.

    Layout:
        q, k: [B, H, T, K]
        v:    [B, H, T, V]
        g:    [B, T, H]
        beta: [B, T, H]
    Returns:
        o: [B, T, H, V]
        final_state: always None (no recurrent-state support)
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, "
            f"v current type is {v.dtype}; they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use float16/bfloat16.")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("g and beta must be rank-3 tensors with shape [B, T, H].")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors with shape [B, H, T, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"q/k/v shape prefixes must match, got {q.shape}, {k.shape}, {v.shape}.")
    if g.shape != beta.shape:
        raise ValueError(f"g and beta shapes must match, got {g.shape} and {beta.shape}.")
    if g.shape[0] != q.shape[0] or g.shape[1] != q.shape[2] or g.shape[2] != q.shape[1]:
        raise ValueError(
            "Expected q/k/v in [B, H, T, D] and g/beta in [B, T, H]; " f"got q={tuple(q.shape)}, g={tuple(g.shape)}."
        )

    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    if cu_seqlens is not None:
        cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using cu_seqlens. "
                "Please flatten variable-length inputs before processing."
            )
    else:
        cu_seqlens_list = None
        chunk_indices = None
        chunk_indices_list = None

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
        initial_state,
        output_final_state,
    )
    return o, final_state


# ---------------------------------------------------------------------------
# NPU replacements for fla.modules classes
# ---------------------------------------------------------------------------


class _CPConvWithState(torch.autograd.Function):
    """Causal conv with a cross-rank initial state (CP).

    Reference semantics (``fla/modules/causal_conv1d/cp.py:88-104``):

    * the first rank injects nothing (it still joins the collective);
    * only ``min(W-1, pre_num_conv_tokens)`` rows of the previous rank's tail are
      real history for this shard;
    * only the FIRST document of a contiguous shard can continue a previous rank.

    The AscendC kernel mutates its state buffer and its backward wants a
    ``[B, W, dim]`` state, so a snapshot is taken before the forward.
    """

    @staticmethod
    def forward(ctx, x, weight, H, cu_seqlens, cp_context):
        import torch.distributed as dist

        if x.dim() != 3:
            raise NotImplementedError(
                "fla_npu CP conv expects x of shape (batch, seqlen, dim)."
            )

        op_weight = weight.transpose(-1, -2).contiguous()
        w_width, dim = op_weight.shape
        pad = w_width - 1

        # The kernel validates queryStartLoc against the token count it is given,
        # so it must describe THIS shard, not the global packed stream.  Prefer
        # the rank-local boundaries carried by the CP context.
        _ctx_cu = getattr(cp_context, "cu_seqlens", None) if cp_context is not None else None
        if _ctx_cu is not None:
            _local_cu = [int(v) for v in (_ctx_cu.tolist() if hasattr(_ctx_cu, "tolist") else _ctx_cu)]
        elif cu_seqlens is not None:
            _local_cu = [int(v) for v in (cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens)]
        else:
            _local_cu = None
        if _local_cu is not None and _local_cu[-1] != int(x.shape[1]):
            _local_cu = [0, int(x.shape[1])]
        if os.environ.get("VIME_CP_CONV_DIAG") == "1":
            _r = dist.get_rank(cp_context.group) if getattr(cp_context, "group", None) is not None else 0
            print(f"[cpconv] rank={_r} T_local={int(x.shape[1])} "
                  f"given_cu={None if cu_seqlens is None else [int(v) for v in (cu_seqlens.tolist() if hasattr(cu_seqlens, 'tolist') else cu_seqlens)]} "
                  f"ctx_cu={_ctx_cu is not None and [int(v) for v in (_ctx_cu.tolist() if hasattr(_ctx_cu, 'tolist') else _ctx_cu)]} "
                  f"chosen={_local_cu}", flush=True)
        cu_seqlens = (
            torch.tensor(_local_cu, dtype=torch.int32, device=x.device) if _local_cu is not None else None
        )
        op_x, query_start_loc, is_varlen = _flatten_varlen_x(x, cu_seqlens)
        num_sequences = len(query_start_loc) - 1 if query_start_loc is not None else int(x.shape[0])
        qsl_device = (
            cu_seqlens.to(device=op_x.device, dtype=torch.int32).contiguous()
            if cu_seqlens is not None
            else None
        )

        group = None if cp_context is None else cp_context.group
        rank = dist.get_rank(group) if group is not None else 0
        world = dist.get_world_size(group) if group is not None else 1
        ctx.group = group
        ctx.rank = rank
        ctx.world = world
        ctx.pad = pad

        tail = x.detach()[:, -pad:, :].contiguous()
        if world > 1:
            gathered = [torch.zeros_like(tail) for _ in range(world)]
            dist.all_gather(gathered, tail, group=group)
            prev = gathered[rank - 1] if rank > 0 else torch.zeros_like(tail)
        else:
            prev = torch.zeros_like(tail)

        state = torch.zeros(num_sequences, w_width, dim, dtype=x.dtype, device=x.device)
        # `FLACPContext` owns both fields and computes `pre_num_conv_tokens` as
        # `max(0, shard_global_start - first_doc_global_start)`, i.e. how many
        # tokens of the containing document precede this shard.  Guessing it
        # (for example defaulting to the kernel width) would splice one
        # document's tail onto a shard that actually starts a new one, so the
        # fields are treated as required rather than defaulted.
        try:
            is_first_rank = bool(cp_context.is_first_rank)
            pre_conv = int(cp_context.pre_num_conv_tokens)
        except AttributeError as exc:
            raise RuntimeError(
                "CP short convolution needs a FLACPContext with both "
                "`is_first_rank` and `pre_num_conv_tokens`; the supplied "
                f"{type(cp_context).__name__} is missing one of them."
            ) from exc
        valid_len = min(pad, pre_conv) if world > 1 else 0
        # The AscendC prefill entry (run_mode=0) only *writes* conv_states; it
        # does not consume them as a prefix history (verified: four different
        # state layouts produced identical outputs).  Splice the real history
        # into the input instead and drop the prefix outputs.
        ctx.window_mode = True
        ctx.window_len = 0
        if valid_len > 0 and not is_first_rank:
            if x.shape[0] != 1:
                raise ValueError("CP prefix windows require one packed token stream.")
            hist = prev[:, -valid_len:, :]
            x_win = torch.cat((hist, x), dim=1)
            base = query_start_loc if query_start_loc is not None else [0, x.shape[1]]
            window_cu = [0] + [int(c) + valid_len for c in base[1:]]
            cu_win = torch.tensor(window_cu, dtype=torch.int32, device=x.device)
            y_full = ascendc_causal_conv1d_fn(
                x_win.flatten(0, 1), op_weight, None, state,
                query_start_loc=cu_win, has_initial_state=None, activation="silu",
                pad_slot_id=-1, null_block_id=None, head_num=H,
            )
            if H:
                if y_full.ndim == 3:
                    y_full = y_full.unsqueeze(0)
                if y_full.shape != (1, H, x_win.shape[1], dim // H):
                    raise RuntimeError(f"Unexpected window conv output {tuple(y_full.shape)}")
                y = y_full[:, :, valid_len:, :]
            else:
                y = y_full.reshape_as(x_win)[:, valid_len:, :]
            ctx.save_for_backward(x_win, op_weight)
            ctx.conv_states = None  # history is already in x_win
            ctx.query_start_loc = window_cu
            ctx.is_varlen = True
            ctx.head_num = H
            ctx.window_len = valid_len
            return y
        ctx.injected = valid_len  # backward must only send these slots back
        ctx.prev_tail = (
            prev.reshape(x.shape[0], -1, x.shape[-1])[:, -valid_len:, :].detach().clone()
            if valid_len > 0
            else None
        )
        if (not is_first_rank) and valid_len > 0:
            state[0, -valid_len:, :] = prev.reshape(-1, dim)[-valid_len:]
        state_snapshot = state.clone()

        y = ascendc_causal_conv1d_fn(
            op_x,
            op_weight,
            None,
            state,
            query_start_loc=qsl_device,
            # state holds the previous rank's tail; without this flag the kernel
            # ignores the buffer and the first W-1 tokens of every non-first
            # rank are convolved from zeros (measured: first3 error 1.7e+00).
            has_initial_state=None,
            activation="silu",
            pad_slot_id=-1,
            null_block_id=None,
            head_num=H,
        )
        if is_varlen:
            y = y.unsqueeze(0)

        ctx.save_for_backward(x, op_weight)
        ctx.conv_states = None if ctx.window_mode else state_snapshot
        ctx.query_start_loc = query_start_loc
        ctx.is_varlen = is_varlen
        ctx.head_num = H
        return y

    @staticmethod
    def backward(ctx, dy):
        import torch.distributed as dist

        x, op_weight = ctx.saved_tensors
        op_x = x.reshape(-1, x.shape[-1]).contiguous() if ctx.is_varlen and x.ndim == 3 else x.contiguous()
        if ctx.window_len:
            # Forward discarded the prefix outputs. Their adjoints are zero.
            time_axis = 2 if ctx.head_num else 1
            prefix_shape = list(dy.shape)
            prefix_shape[time_axis] = ctx.window_len
            dy = torch.cat((dy.new_zeros(prefix_shape), dy), dim=time_axis)
        if ctx.is_varlen and (dy.ndim == 4 or (ctx.window_mode and not ctx.head_num)):
            op_dy = dy.squeeze(0).contiguous()
        else:
            op_dy = dy.contiguous()

        qsl_device = None
        if ctx.query_start_loc is not None:
            qsl_device = torch.tensor(ctx.query_start_loc, dtype=torch.int32, device=op_x.device).contiguous()

        preactivation = ascendc_causal_conv1d_fn(
            op_x,
            op_weight,
            None,
            None if ctx.conv_states is None else ctx.conv_states.clone(),
            query_start_loc=qsl_device,
            has_initial_state=None,
            activation=None,
            pad_slot_id=-1,
            null_block_id=None,
            head_num=ctx.head_num,
        )
        op_y = (
            preactivation.squeeze(0).contiguous()
            if ctx.is_varlen and preactivation.ndim == 4
            else preactivation.contiguous()
        )

        dx, dw, _db, dh0 = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=op_y,
            weight=op_weight,
            dy=op_dy,
            initial_state=ctx.conv_states,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=1,
            input_layout="NTD" if ctx.is_varlen else "BNSD",
        )
        dx = dx.reshape_as(x)
        dw = dw.transpose(0, 1).contiguous()

        # Every rank joins, including document starts with no prefix.
        # Align the newest history row with the last row of the buffer.
        h = ctx.window_len
        local_dx = dx[:, h:, :].contiguous()
        if ctx.world > 1 and ctx.pad:
            halo = dx.new_zeros((dx.shape[0], ctx.pad, dx.shape[-1]))
            if h:
                halo[:, -h:, :].copy_(dx[:, :h, :])
            received = [torch.empty_like(halo) for _ in range(ctx.world)]
            dist.all_gather(received, halo, group=ctx.group)
            if ctx.rank + 1 < ctx.world:
                n = min(ctx.pad, local_dx.shape[1])
                if n:
                    local_dx[:, -n:, :].add_(received[ctx.rank + 1][:, -n:, :])
        return local_dx, dw, None, None, None
class ShortConvolution(nn.Module):
    """NPU replacement for ``fla.modules.ShortConvolution``.

    Routes the depthwise causal convolution through the fla-npu AscendC
    kernel via an autograd Function.  Unlike the ``fla`` version this
    carries no recurrent state: the training path never uses a conv state.

    The weight is stored with the ``nn.Conv1d`` layout
    ``(hidden_size, 1, kernel_size)`` so that HuggingFace checkpoints load
    directly.  ``forward`` squeezes the in_channels axis to obtain
    ``(hidden_size, kernel_size)`` = ``(dim, width)`` for the AscendC kernel.

    ``head_num`` selects the kernel's output layout and matches the rollout
    (vllm-ascend) path.  When ``head_num > 0`` the kernel returns a
    head-first ``[H, T, D]`` tensor (``[H, T, D//H]`` per head) and the SiLU
    is applied inside the kernel; when ``head_num == 0`` the output stays
    flat ``[B, T, D]`` with SiLU in Python.  The two paths use different
    AscendC tiling/SiLU implementations and diverge under bfloat16, so the
    caller must pass the same ``head_num`` the rollout uses to keep the
    training and rollout logprobs aligned.

    ``bias`` is accepted for signature compatibility with the ``fla``
    version but must be False: the Qwen3.5/3.6 GDN conv is bias-free in the
    HF checkpoint and the bias plumbing is not implemented.
    """

    def __init__(self, hidden_size: int, bias: bool = False, kernel_size: int = 4, head_num: int = 0):
        super().__init__()
        if bias:
            raise NotImplementedError(
                "ShortConvolution on NPU does not support bias; the Qwen3.5/3.6 GDN conv is bias-free."
            )
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.head_num = head_num
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        nn.init.normal_(self.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cp_context=None,
    ):
        # activation="silu" is applied inside the AscendC kernel, matching
        # the rollout (vllm-ascend gdn.py) path; head_num is forwarded
        # verbatim so the kernel picks the same head-first / flat layout as
        # the rollout: head_num>0 -> [H, T, D] head-first, head_num==0 -> flat.
        if cp_context is not None:
            # The kernel needs a grad-free recurrent state buffer, so the whole
            # conv (state exchange included) runs inside our own Function.
            out = _CPConvWithState.apply(
                x,
                self.weight.squeeze(1),
                self.head_num,
                cu_seqlens,
                cp_context,
            )
            return out, None

        out = causal_conv1d_ascendc(
            x,
            weight=self.weight.squeeze(1),
            H=self.head_num,
            cu_seqlens=cu_seqlens,
        )
        return out, None


class QwenStyleRMSNormGated(nn.Module):
    """RMSNorm + gated activation on NPU (replacement for ``fla.modules.FusedRMSNormGated``).

    Mirrors the upstream ``fla`` constructor signature
    ``(hidden_size, eps, activation, device, dtype)`` so it is a drop-in
    replacement, but only ``hidden_size`` and ``eps`` affect the computation.
    The gate activation is always SiLU, matching the Qwen3.5 reference path;
    any other requested activation is rejected instead of silently ignored.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str | None = "silu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if activation not in (None, "", "silu", "swish"):
            raise ValueError(f"QwenStyleRMSNormGated only supports silu, got {activation!r}.")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        inp_dtype = hidden_states.dtype
        hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, self.variance_epsilon)[0]
        hidden_states = hidden_states * F.silu(gate.to(hidden_states.dtype))
        return hidden_states.to(inp_dtype)
