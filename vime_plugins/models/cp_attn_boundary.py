"""Layout boundary for dense attention when the CP main layout is contiguous.

With ``--allgather-cp`` the packed token stream is sliced contiguously, which is
the layout fla's chunkwise CP kernels consume natively -- so the (majority) GDN
layers need no shuffling at all.  Transformer-Engine's ring attention, however,
only understands megatron's per-document zigzag partition.

The wrapper is a *subclass* of the configured attention module rather than a
containing ``nn.Module``: submodule names (and therefore the HF <-> Megatron
weight mapping) stay exactly as they were.
"""

from copy import copy

from megatron.core.transformer.spec_utils import ModuleSpec, import_module


def _make_boundary_cls(base_cls, args):
    class CPAttnBoundary(base_cls):
        """Contiguous -> zigzag -> ring attention -> contiguous."""

        def forward(self, hidden_states, **kwargs):
            packed = kwargs.get("packed_seq_params")
            if packed is None or not getattr(args, "allgather_cp", False):
                return super().forward(hidden_states, **kwargs)

            from megatron.core import mpu

            group = mpu.get_context_parallel_group()
            if group is None or group.size() == 1:
                return super().forward(hidden_states, **kwargs)

            from .gdn.cp_layout import _skip_convert, contiguous_to_zigzag, zigzag_to_contiguous

            if _skip_convert():
                return super().forward(hidden_states, **kwargs)

            cu = getattr(packed, "cu_seqlens_q_cpu", None)
            if cu is None:
                cu = packed.cu_seqlens_q.tolist()

            alignment = 2 * group.size()
            padded = [0]
            for start, end in zip(cu[:-1], cu[1:], strict=True):
                length = end - start
                padded.append(padded[-1] + length + (-length % alignment))

            padded_cu = None
            if tuple(padded) != tuple(cu):
                padded_cu = padded
                params = copy(packed)
                offsets = packed.cu_seqlens_q.new_tensor(padded)
                params.cu_seqlens_q = params.cu_seqlens_kv = offsets
                if getattr(packed, "cu_seqlens_q_padded", None) is not None:
                    params.cu_seqlens_q_padded = offsets
                if getattr(packed, "cu_seqlens_kv_padded", None) is not None:
                    params.cu_seqlens_kv_padded = offsets
                if hasattr(params, "cu_seqlens_q_cpu"):
                    params.cu_seqlens_q_cpu = tuple(padded)
                maximum = max(b - a for a, b in zip(padded[:-1], padded[1:], strict=True))
                before_q = getattr(packed, "max_seqlen_q", None)
                params.max_seqlen_q = params.max_seqlen_kv = maximum
                kwargs["packed_seq_params"] = params
            # --- SP entry: caller hands [T, B, H] in SP (TP-split) layout ---
            from megatron.core import tensor_parallel as _tp

            _sp = bool(getattr(args, "sequence_parallel", False))
            _tp_group = mpu.get_tensor_model_parallel_group()
            entry_shape = tuple(hidden_states.shape)
            sp_tbh = hidden_states
            if _sp:
                sp_tbh = _tp.gather_from_sequence_parallel_region(
                    sp_tbh, tensor_parallel_output_grad=False, group=_tp_group
                )
            full_rows = int(sp_tbh.shape[0])

            # CP redistribution needs the full CP shard -> zigzag in [B, T, H].
            zigzag_full_bth = contiguous_to_zigzag(
                sp_tbh.transpose(0, 1), cu, group, padded_cu=padded_cu
            )
            # The zigzag slots are padded out to 2*cp alignment, so the rotary
            # table (built from the real position ids) must be padded too or
            # aclnnRotaryPositionEmbedding rejects the shape mismatch
            # (cos [175,1,1,D] vs q [176,1,H,D]).  Pad rows are dropped later.
            # RoPE consumes the table per document (`_get_freqs_on_this_cp_rank`
            # slices each doc from position 0 and needs `cp_size * per_rank_rows`
            # rows), so the target is the widest *padded* document, not this
            # rank's total packed rows. Comparing against the total skipped the
            # pad entirely (516 - 767 < 0) while the widest doc still needed 768.
            rot = kwargs.get("rotary_pos_emb")
            if rot is not None:
                kwargs["rotary_pos_emb"] = pad_rotary_to_widest_document(rot, padded)
            # --- hand the SP layout back to the wrapped attention ---
            zigzag_sp_tbh = zigzag_full_bth
            if _sp:
                zigzag_sp_tbh = _tp.scatter_to_sequence_parallel_region(
                    zigzag_full_bth.transpose(0, 1).contiguous(), group=_tp_group
                ).transpose(0, 1).contiguous()

            out, bias = super().forward(zigzag_sp_tbh.transpose(0, 1).contiguous(), **kwargs)

            # --- exit: SP -> gather -> zigzag->contiguous -> scatter back to SP ---
            out_tbh = out
            if _sp:
                out_tbh = _tp.gather_from_sequence_parallel_region(
                    out_tbh, tensor_parallel_output_grad=False, group=_tp_group
                )
            out_contig = zigzag_to_contiguous(
                out_tbh.transpose(0, 1), cu, group, padded_cu=padded_cu
            )
            result = out_contig.permute(1, 0, 2).contiguous()
            if _sp:
                result = _tp.scatter_to_sequence_parallel_region(result, group=_tp_group)
                if not globals().get("_ATTN_SP_DIAG"):
                    globals()["_ATTN_SP_DIAG"] = True
                    print(f"[sp] attn: entry={entry_shape} full={full_rows} "
                          f"zigzag={tuple(zigzag_full_bth.shape)} "
                          f"sp_in={tuple(zigzag_sp_tbh.shape)} "
                          f"sp_out={tuple(zigzag_sp_tbh.shape)} "
                          f"return={tuple(result.shape)}", flush=True)
            assert tuple(result.shape) == entry_shape, (
                f"CPAttnBoundary shape mismatch: entry={entry_shape} result={tuple(result.shape)}"
            )
            return result, bias

    CPAttnBoundary.__name__ = f"CPAttnBoundary{base_cls.__name__}"
    CPAttnBoundary.__qualname__ = CPAttnBoundary.__name__
    return CPAttnBoundary


def wrap_attention_spec(inner_spec, args):
    """Re-spec the configured attention module with the layout-converting forward."""
    base = inner_spec.module
    if isinstance(base, str):
        base = import_module(base)
    params = dict(getattr(inner_spec.params, "copy", lambda: dict(inner_spec.params))())
    spec = ModuleSpec(module=_make_boundary_cls(base, args), params=params)
    submodules = getattr(inner_spec, "submodules", None)
    if submodules is not None:
        spec.submodules = submodules
    return spec

def widest_padded_document(padded_cu):
    """Width of the widest document after `2*cp` alignment padding."""
    return max(b - a for a, b in zip(padded_cu[:-1], padded_cu[1:], strict=True))


def pad_rotary_to_widest_document(rotary, padded_cu):
    """Extend the RoPE table so every padded document can be encoded.

    `_get_freqs_on_this_cp_rank` slices each document from position 0 and needs
    `cp_size * per_rank_rows` rows, i.e. as many rows as the widest *padded*
    document. The table is built earlier from the unpadded maximum, so an
    aligned boundary can leave it one row short and the kernel rejects the
    shape. Comparing against the rank's total packed rows instead of the widest
    document makes `extra` negative and silently skips the pad.

    Only trailing pad rows are appended (a copy of the last row); every real
    position keeps its own row.
    """
    target = widest_padded_document(padded_cu)
    if rotary.shape[0] >= target:
        return rotary
    import torch

    extra = target - rotary.shape[0]
    return torch.cat([rotary, rotary[-1:].expand(extra, *rotary.shape[1:])], dim=0)
