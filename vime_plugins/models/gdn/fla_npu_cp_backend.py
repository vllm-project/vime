"""Route FLA's Ascend CP pre-process / merge kernels to fla_npu AscendC ops.

FLA dispatches CP state handling through ``TritonAscendCPBackend``
(``@register('cp')``), whose two entry points end in
``fla/ops/cp/backends/triton_ascend/chunk_delta_h.py``:

    chunk_gated_delta_rule_fwd_h_pre_process_npu  -> launch_fwd -> _launch_preprocess_kernel(...)
    chunk_gated_delta_rule_bwd_dhu_pre_process_npu -> launch_bwd -> _launch_preprocess_kernel(...)

The orchestration (cu_seqlens window, chain ordering, multi-seq, grid-axis
chunking) must stay; only the *kernel* should move to AscendC.  So this module
replaces the two leaf points:

    _launch_preprocess_kernel(...)      -> npu_pre_process_{fwd,bwd}_kernel_merged
    merge_fwd_bwd_kernel_npu[grid](...) -> npu_merge_fwd_bwd_kernel

Layout note (learned the hard way): in this backend ``k``/``w``/``u`` are
``[B, T, H, D]`` and ``g`` is already ``[B, HV, T]`` (see ``_contig_gate``).
The AscendC kernels want ``[B, H, T, D]``, hence the transposes below.
"""

from __future__ import annotations

import torch

_PATCHED = False


def _npu():
    from fla_npu.ops.ascendc._aclnn_ctypes import (
        npu_merge_fwd_bwd_kernel,
        npu_pre_process_bwd_kernel_merged,
        npu_pre_process_fwd_kernel_merged,
    )

    return (
        npu_pre_process_fwd_kernel_merged,
        npu_pre_process_bwd_kernel_merged,
        npu_merge_fwd_bwd_kernel,
    )


def _to_bh(kw, key):
    t = kw.get(key)
    if t is None or t.dim() != 4:
        return t
    return t.transpose(1, 2).contiguous()


def _install_preprocess(module):
    fwd_op, bwd_op, _ = _npu()
    orig = module._launch_preprocess_kernel
    fwd_kernel = getattr(module, "pre_process_fwd_kernel_merged_npu", None)
    bwd_kernel = getattr(module, "pre_process_bwd_kernel_merged_npu", None)

    def _launch_preprocess_kernel(kernel, *, nv_chunks, nh_total, kernel_kwargs):
        kw = kernel_kwargs
        is_fwd = kernel is fwd_kernel
        is_bwd = kernel is bwd_kernel
        if not (is_fwd or is_bwd):
            return orig(kernel, nv_chunks=nv_chunks, nh_total=nh_total, kernel_kwargs=kernel_kwargs)

        # The AscendC ops consume the whole tensor, so only launch once and let
        # the caller's v/nh chunk loops become no-ops.
        if kw.get("V_OFFSET", 0) != 0 or kw.get("NH_OFFSET", 0) != 0:
            return
        if kw.get("PART", 0) == 2:
            return

        cu = kw.get("cu_seqlens")
        cu = [int(x) for x in cu] if cu is not None else None

        if is_fwd:
            # IEEE-FP32 affine chain, matching FLA's Triton-Ascend backend.
            out = fwd_op(
                _to_bh(kw, "k"),
                _to_bh(kw, "w"),
                _to_bh(kw, "u"),
                g=kw.get("g"),
                gk=kw.get("gk"),
                bg=kw.get("bg"),
                v=None,
                cu_seqlens=cu,
                chunk_size=int(kw.get("BT", 64)),
            )
            hm = kw["hm"]
            hm.copy_(out.reshape(hm.shape).to(hm.dtype))
            return

        out = bwd_op(
            _to_bh(kw, "q"),
            _to_bh(kw, "k"),
            _to_bh(kw, "w"),
            _to_bh(kw, "do"),
            _to_bh(kw, "dv"),
            float(kw.get("scale") or 1.0),
            int(kw.get("BT", 64)),
            g=kw.get("g"),
            gk=kw.get("gk"),
            cu_seqlens=cu,
        )
        dhm = kw["dhm"] if kw.get("dhm") is not None else kw["hm"]
        dhm.copy_(out.reshape(dhm.shape).to(dhm.dtype))

    module._launch_preprocess_kernel = _launch_preprocess_kernel
    return module


class _MergeShim:
    """Stand-in for ``merge_fwd_bwd_kernel_npu`` (ignores the Triton grid)."""

    def __getitem__(self, grid):  # noqa: D105 - mimic kernel[grid]
        return self

    def __call__(self, **kw):  # noqa: D102
        _, _, merge_op = _npu()
        merge_op(
            kw["h"],
            kw["ag_hm"],
            kw["pre_or_post_num_ranks"],
            kw["rank"],
            forward=bool(kw.get("FORWARD", True)),
            state_v_first=bool(kw.get("STATE_V_FIRST", False)),
        )
        return None


def install(verbose: bool = True) -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import fla.ops.cp.backends.triton_ascend.chunk_delta_h as cp_npu
    except ImportError as exc:  # pragma: no cover
        print(f"[fla_npu_cp_backend] import failed: {exc}")
        return False

    _install_preprocess(cp_npu)
    cp_npu.merge_fwd_bwd_kernel_npu = _MergeShim()

    _PATCHED = True
    if verbose:
        print("[fla_npu_cp_backend] patched: _launch_preprocess_kernel, merge_fwd_bwd_kernel_npu")
    return True
