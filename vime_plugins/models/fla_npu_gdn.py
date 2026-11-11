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
    ):
        op_weight = weight.transpose(-1, -2).contiguous()
        width, dim = op_weight.shape
        op_x, query_start_loc, is_varlen = _flatten_varlen_x(x, cu_seqlens)
        num_sequences = len(query_start_loc) - 1 if query_start_loc is not None else int(x.shape[0])
        # No initial_state support: the conv always starts from a zero state.
        conv_states = torch.zeros(num_sequences, width - 1, dim, dtype=x.dtype, device=x.device)
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
            None,  # bias: the Qwen3.5/3.6 conv is bias-free
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

        dx, dw, _, _ = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=op_y,
            weight=op_weight,
            dy=op_dy,
            initial_state=None,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=1,  # silu
            input_layout="NTD" if ctx.is_varlen else "BNSD",
        )
        dx = dx.reshape_as(x)
        dw = dw.transpose(0, 1).contiguous()
        return dx, dw, None, None


def causal_conv1d_ascendc(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    return AscendCCausalConv1dFunction.apply(x, weight, H, cu_seqlens)


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
):
    chunk_list = _chunk_list(chunk_indices_list, chunk_size)
    # The composite AscendC kernel runs the whole forward (cumsum, kkt,
    # solve_tri, WY repr, recurrent, output) and returns the cumsum'd g and A
    # needed by the backward pass.
    o, _, g, A, *_ = ascendc_chunk_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_list,
        scale=scale,
        disable_recompute=True,
        return_intermediate_states=False,
        use_qk_l2norm_in_kernel=False,
    )
    return g, o, A


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
):
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

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
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

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

    # No recurrent-state support: h0 / dht stay None, matching the forward,
    # which never takes an initial_state and never outputs a final_state.
    dh, _, dv = ascendc_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
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
    return dq, dk, dv, db, dg


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
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
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
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg = flash_chunk_gated_delta_rule_bwd(
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

    o = ChunkGatedDeltaRuleFunction.apply(
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
    )
    return o, None


# ---------------------------------------------------------------------------
# NPU replacements for fla.modules classes
# ---------------------------------------------------------------------------


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

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None):
        # activation="silu" is applied inside the AscendC kernel, matching
        # the rollout (vllm-ascend gdn.py) path; head_num is forwarded
        # verbatim so the kernel picks the same head-first / flat layout as
        # the rollout: head_num>0 -> [H, T, D] head-first, head_num==0 -> flat.
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
