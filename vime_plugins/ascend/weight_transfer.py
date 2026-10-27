"""Ascend's vLLM trainer init-info factories.

Named by NPUAccelerator.weight_transfer_init_info_qualname() and imported
only when a transfer is actually being set up, so neither selecting the
backend nor describing a transport pulls in vllm_ascend.

These are functions rather than the init-info classes themselves because the
classes live behind vLLM's plugin loading, which a trainer process has not
necessarily done yet.
"""

from __future__ import annotations

from typing import Any


def _load_vllm_ascend() -> None:
    import torch_npu  # noqa: F401  (registers the NPU device with torch)
    from vllm.plugins import load_general_plugins

    # Engine workers load general plugins during vLLM startup; a trainer
    # process never went through that path, so its vllm_ascend engines are
    # not registered yet.
    load_general_plugins()


def npu_ipc_trainer_init_info(**kwargs: Any) -> Any:
    _load_vllm_ascend()
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

    return NPUIPCTrainerInitInfo(**kwargs)


def hccl_trainer_init_info(**kwargs: Any) -> Any:
    _load_vllm_ascend()
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

    return HCCLTrainerInitInfo(**kwargs)
