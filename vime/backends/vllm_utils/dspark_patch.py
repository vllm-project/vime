# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime patch for vLLM DFlash/DSpark meta-device tensor bug.

Applies the fix from vllm-project/vllm#55076 at import time so that
DSpark/DFlash draft models survive IPC engine weight sync without crashing
on meta-device tensors in ``_build_context_kv_buffers``.

Remove this module once vLLM #55076 is merged and the base image includes it.
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_vllm_dspark_meta_device():
    """Monkeypatch ``qwen3_dflash._build_context_kv_buffers`` and
    ``_build_fused_kv_buffers`` to materialize meta-device tensors to CUDA
    before concatenation.

    This is a temporary workaround for vLLM versions that do not yet include
    the upstream fix (vllm-project/vllm#55076).
    """
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import torch

        from vllm.model_executor.models import qwen3_dflash as mod

        def _is_meta(t):
            return t is not None and t.device.type == "meta"

        def _resolve_device(*tensors):
            for t in tensors:
                if t is not None and t.device.type != "meta":
                    return t.device
            return torch.device("cuda", torch.cuda.current_device())

        def _build_context_kv_buffers_patched(self, layers_attn, has_bias):
            self._hidden_norm_weight = self.hidden_norm.weight.data

            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn] if has_bias else []
            k_norm_weights = [a.k_norm.weight.data for a in layers_attn]
            needed = [self._hidden_norm_weight, *kv_weights, *k_norm_weights, *kv_biases]

            if any(_is_meta(t) for t in needed):
                if hasattr(self, "_fused_kv_weight"):
                    logger.warning_once(
                        "Skipping DFlash fused KV rebuild: some attention weights are still on the meta device after a partial load_weights (e.g. IPC weight sync). Keeping previously built buffers."
                    )
                    return
                raise RuntimeError(
                    "DFlash fused KV build found attention weights on the meta device and no previous CUDA buffers exist. This usually means load_weights did not materialize draft qkv_proj/k_norm."
                )

            device = _resolve_device(*needed)
            self._hidden_norm_weight = self._hidden_norm_weight.to(device)
            self._fused_kv_weight = torch.cat([w.to(device) for w in kv_weights], dim=0)
            if has_bias:
                self._fused_kv_bias = torch.cat([b.to(device) for b in kv_biases], dim=0)
            else:
                self._fused_kv_bias = None
            self._k_norm_weights = torch.stack([w.to(device) for w in k_norm_weights], dim=0).contiguous()

        def _build_fused_kv_buffers_patched(self):
            layers_attn = [layer.self_attn for layer in self.layers]
            attn0 = layers_attn[0]
            has_bias = attn0.qkv_proj.bias is not None

            self._build_context_kv_buffers(layers_attn, has_bias)

            self._rope_head_size = attn0.rotary_emb.head_size
            cos_sin_cache = attn0.rotary_emb.cos_sin_cache
            if cos_sin_cache is not None and cos_sin_cache.device.type == "meta":
                compute = getattr(attn0.rotary_emb, "_compute_cos_sin_cache", None)
                if compute is not None:
                    with torch.device("cpu"):
                        cos_sin_cache = compute()
                else:
                    cos_sin_cache = torch.empty(
                        cos_sin_cache.shape,
                        device="cuda",
                        dtype=cos_sin_cache.dtype,
                    )
                cos_sin_cache = cos_sin_cache.to(device=_resolve_device(), dtype=cos_sin_cache.dtype)
                attn0.rotary_emb.cos_sin_cache = cos_sin_cache
            self._rope_cos_sin_cache = cos_sin_cache
            self._rope_is_neox = attn0.rotary_emb.is_neox_style

            self._num_attn_layers = len(layers_attn)
            self._kv_size = attn0.kv_size
            self._head_dim = attn0.head_dim
            self._num_kv_heads = attn0.num_kv_heads
            self._rms_norm_eps = attn0.q_norm.variance_epsilon
            self._attn_layers = [layer.self_attn.attn for layer in self.layers]

        # Check if already patched or already fixed upstream
        orig = mod.DFlashQwen3Model._build_context_kv_buffers
        if "_patched" not in getattr(orig, "__qualname__", ""):
            mod.DFlashQwen3Model._build_context_kv_buffers = _build_context_kv_buffers_patched
            mod.DFlashQwen3Model._build_fused_kv_buffers = _build_fused_kv_buffers_patched
            logger.info("Applied DFlash/DSpark meta-device patch (vLLM#55076)")

    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to apply DFlash meta-device patch: {e}")
