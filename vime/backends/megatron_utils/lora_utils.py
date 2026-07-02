"""LoRA helpers for Megatron actor training and vLLM adapter serving."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from argparse import Namespace
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_ALL_LINEAR_HF_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_lora_enabled(args: Namespace) -> bool:
    return args.lora_rank > 0


def parse_target_modules(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value in ("all", "all-linear", "all_linear"):
            return list(_ALL_LINEAR_HF_MODULES)
        return [item.strip() for item in value.split(",") if item.strip()]
    modules: list[str] = []
    for item in value:
        modules.extend(parse_target_modules(item))
    return modules


def normalize_target_modules(
    target_modules: str | Sequence[str], exclude_modules: str | Sequence[str] | None = None
) -> list[str]:
    modules = parse_target_modules(target_modules)
    exclude = set(parse_target_modules(exclude_modules))
    return [module for module in modules if module not in exclude]


def convert_target_modules_to_megatron(hf_modules: str | Sequence[str]) -> list[str]:
    modules = parse_target_modules(hf_modules)
    out: list[str] = []
    for module in modules:
        megatron_module = _STANDARD_LORA_HF_TO_MEGATRON.get(module, module)
        if megatron_module not in out:
            out.append(megatron_module)
    return out


def create_lora_instance(args: Namespace):
    """Create a Megatron-Bridge LoRA object.

    This mirrors the reference implementation's Bridge-based approach while
    keeping vime's rollout backend vLLM-native.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_cls = CanonicalLoRA if args.lora_type.lower() == "canonical_lora" else LoRA
    # --exclude-modules is already applied to args.target_modules during arg
    # validation; Megatron-Bridge asserts that exclude_modules stays empty
    # whenever target_modules is set, so only target_modules is passed here.
    target_modules = convert_target_modules_to_megatron(args.target_modules)

    lora = lora_cls(
        target_modules=target_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    logger.info(
        "Created %s: rank=%s alpha=%s dropout=%s target_modules=%s",
        lora_cls.__name__,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        target_modules,
    )
    return lora


def lora_runtime_dir(args: Namespace, step: int) -> Path:
    root = Path(args.save or tempfile.gettempdir())
    return root / "vime_lora_runtime" / f"step_{step}"


@lru_cache(maxsize=1)
def _export_bridge(hf_checkpoint: str):
    from megatron.bridge import AutoBridge

    import vime_plugins.megatron_bridge  # noqa: F401
    from vime.utils import megatron_bridge_utils

    return megatron_bridge_utils.patch_auto_bridge_hf_config(
        AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)
    )


def infer_hf_target_modules(adapter_weight_names: Iterable[str]) -> list[str]:
    """Derive PEFT target_modules from exported adapter weight names.

    Megatron trains fused projections (e.g. linear_qkv), so the exported HF
    tensors can cover more modules than the user passed via --target-modules;
    the config must declare what the weights actually contain.
    """
    modules: set[str] = set()
    for name in adapter_weight_names:
        prefix, sep, _ = name.partition(".lora_A")
        if not sep:
            prefix, sep, _ = name.partition(".lora_B")
        if sep:
            modules.add(prefix.rsplit(".", 1)[-1])
    return sorted(modules)


def save_lora_adapter_for_vllm(model: Sequence[torch.nn.Module], args: Namespace, step: int) -> str:
    """Export the current actor adapter in PEFT format for vLLM runtime loading."""
    from vime.utils import megatron_bridge_utils

    save_dir = lora_runtime_dir(args, step)
    # Save on every node's local rank 0 so that colocated vLLM engines on each
    # node can read the adapter from their own local filesystem. All ranks hold
    # the full adapter state dict after export, so this needs no extra comm.
    is_node_main = int(os.environ.get("LOCAL_RANK", "0")) == 0

    if is_node_main:
        save_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())

    bridge = _export_bridge(args.hf_checkpoint)
    lora_state_dict: dict[str, torch.Tensor] = {}
    with megatron_bridge_utils.patch_megatron_model(model):
        for hf_name, weight, _megatron_name in bridge.export_adapter_weights(
            model,
            cpu=False,
            show_progress=False,
        ):
            lora_state_dict[hf_name] = weight.detach().clone().contiguous().cpu()

    if not lora_state_dict:
        raise RuntimeError("No LoRA adapter weights were exported; is the actor model missing PEFT adapters?")

    if is_node_main:
        torch.save(lora_state_dict, save_dir / "adapter_model.bin")
        config = build_peft_lora_config(args, infer_hf_target_modules(lora_state_dict))
        with open(save_dir / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Saved vLLM LoRA adapter %s with %s tensors", save_dir, len(lora_state_dict))
        # vLLM copies adapter weights into memory at load time, so directories
        # from earlier steps are never read again; drop them to bound disk
        # usage. Best-effort because node-main ranks may share a filesystem.
        for old_dir in save_dir.parent.glob("step_*"):
            if old_dir != save_dir:
                shutil.rmtree(old_dir, ignore_errors=True)

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    return str(save_dir)


def build_peft_lora_config(args: Namespace, target_modules: Sequence[str]) -> dict[str, Any]:
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": list(target_modules),
    }
