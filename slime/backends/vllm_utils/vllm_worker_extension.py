"""vLLM worker extension for vime colocated (IPC) mode.

Passed to ``vllm serve`` via ``--worker-extension-cls`` so that the IPC
engine patch is applied inside every vLLM worker process automatically,
without requiring explicit patching from the trainer side.
"""

from __future__ import annotations


class _VLLMHijack:
    """Applies monkey-patches to vLLM internals required for vime colocated IPC weight sync."""

    @staticmethod
    def hijack() -> None:
        """Patch ``IPCWeightTransferEngine.receive_weights`` to call
        ``monkey_patch_torch_reductions`` before deserialising IPC handles.

        Idempotent – safe to call multiple times.
        """
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

        if getattr(IPCWeightTransferEngine, "_slime_receive_patched", False):
            return

        from slime.backends.megatron_utils.update_weight.torch_patch import monkey_patch_torch_reductions

        _orig = IPCWeightTransferEngine.receive_weights

        def _slime_receive_weights(self, update_info, load_weights, _orig=_orig):
            monkey_patch_torch_reductions()
            _orig(self, update_info, load_weights)

        IPCWeightTransferEngine.receive_weights = _slime_receive_weights
        IPCWeightTransferEngine._slime_receive_patched = True  # type: ignore[attr-defined]


class vLLMColocateWorkerExtension:
    """vLLM worker extension for vime colocated (IPC) weight-sync mode.

    vLLM instantiates this class inside each worker process when
    ``--worker-extension-cls`` is supplied.  ``__new__`` is the earliest
    reliable hook to apply process-wide patches before any weight transfer
    occurs.
    """

    def __new__(cls, **kwargs):
        # Apply the IPC engine patch so every worker process handles
        # CUDA IPC handle deserialisation correctly.
        _VLLMHijack.hijack()
        return super().__new__(cls)
    
    def funny():
        pass
