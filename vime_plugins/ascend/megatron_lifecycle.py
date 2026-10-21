"""Ascend's Megatron startup behavior.

Named by NPUAccelerator.megatron_lifecycle_qualname(). Replaces
vime.backends.megatron_lifecycle for NPU jobs. Importing this module has no
vendor side effects; before_import() is the point where they happen.

Offload is not here: Megatron's buffer-allocation hook and the patched NPU
memory saver handle it, so the shared training and weight-export paths apply
unchanged.
"""

from __future__ import annotations

import importlib
from threading import RLock
from typing import Any

from vime.utils import accelerator

_lock = RLock()
_install_state = "pending"


def _install_attention_override() -> None:
    """Point Megatron's attention at the fused NPU kernel.

    Re-read from the module each time: MegatronAdaptor replaces the class
    attribute whenever it re-applies its features, and importing an
    already-cached module does not reinstall anything.
    """
    attention = importlib.import_module("vime_plugins.ascend.attention_patch")
    attention.DotProductAttention.forward = attention.npu_dot_product_attention_forward


def before_import() -> None:
    """Install the Ascend layer before anything imports Megatron.

    A partial install leaves Megatron half-patched in ways that surface much
    later as wrong numerics, so a second attempt in the same process is
    refused rather than retried.
    """
    global _install_state
    with _lock:
        if _install_state == "done":
            return
        if _install_state != "pending":
            raise RuntimeError("Ascend patch installation did not complete; restart this process")
        _install_state = "installing"

        importlib.import_module("torch_npu")
        selected = accelerator.get_accelerator()
        if selected.name != "npu":
            raise RuntimeError(f"Ascend startup cannot run under the already selected {selected.name!r} accelerator")

        # MindSpeed and torch_memory_saver call the CUDA spelling. Alias it
        # without swallowing allocator failures: an empty_cache that raises is
        # reporting a real device problem.
        torch = importlib.import_module("torch")
        torch.cuda.empty_cache = torch.npu.empty_cache

        # MegatronAdaptor installs its pre-patches on import, which must
        # happen before any Megatron module is imported. The attention
        # override goes on top of them.
        importlib.import_module("megatron_adaptor")
        _install_attention_override()
        _install_state = "done"


def after_args(args: Any) -> None:
    """Re-apply MegatronAdaptor's features against the resolved arguments.

    Its feature set is chosen from arguments that are only complete after
    Megatron has parsed and validated them, so the import-time pass used
    placeholders.
    """
    features_manager = importlib.import_module("megatron_adaptor.features_manager.features_manager").FeaturesManager
    full_args = importlib.import_module("megatron_adaptor.utils.args_utils").get_full_args()
    for key, value in vars(args).items():
        setattr(full_args, key, value)
    features_manager.remove_patches()
    features_manager.apply_features_pre_patches(full_args)
    features_manager.apply_features_patches(full_args)
    # Re-applying can replace attention again.
    _install_attention_override()
