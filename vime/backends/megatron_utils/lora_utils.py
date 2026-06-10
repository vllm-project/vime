"""LoRA helpers for Megatron actor training and vLLM adapter serving."""

from __future__ import annotations

import json
import logging
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from vime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "vime_lora"

_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_MEGATRON_TO_HF_MODULES = {
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
}

_ALL_LINEAR_HF_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_HF_MODULE_NAMES = set(_ALL_LINEAR_HF_MODULES)


def is_lora_enabled(args: Namespace) -> bool:
    return getattr(args, "lora_rank", 0) > 0


def is_lora_weight_name(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name or "lora_A" in name or "lora_B" in name


def is_adapter_param_name(name: str) -> bool:
    return is_lora_weight_name(name) or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


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


def parse_exclude_modules(value: str | Sequence[str] | None) -> list[str]:
    return parse_target_modules(value)


def normalize_target_modules(target_modules: str | Sequence[str], exclude_modules: str | Sequence[str] | None = None) -> list[str]:
    modules = parse_target_modules(target_modules)
    exclude = set(parse_exclude_modules(exclude_modules))
    return [module for module in modules if module not in exclude]


def convert_target_modules_to_megatron(hf_modules: str | Sequence[str]) -> list[str]:
    modules = parse_target_modules(hf_modules)
    out: list[str] = []
    for module in modules:
        megatron_module = _STANDARD_LORA_HF_TO_MEGATRON.get(module, module)
        if megatron_module not in out:
            out.append(megatron_module)
    return out


def convert_target_modules_to_hf(megatron_modules: Sequence[str]) -> list[str]:
    out: list[str] = []
    for module in megatron_modules:
        leaf = module.split(".")[-1]
        mapped = _MEGATRON_TO_HF_MODULES.get(leaf, [leaf])
        for item in mapped:
            if item not in out:
                out.append(item)
    return out


def create_lora_instance(args: Namespace):
    """Create a Megatron-Bridge LoRA object.

    This mirrors the reference implementation's Bridge-based approach while
    keeping vime's rollout backend vLLM-native.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_type = getattr(args, "lora_type", "lora").lower()
    lora_cls = CanonicalLoRA if lora_type == "canonical_lora" else LoRA
    target_modules = convert_target_modules_to_megatron(args.target_modules)
    exclude_modules = convert_target_modules_to_megatron(getattr(args, "exclude_modules", None) or [])

    lora = lora_cls(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    logger.info(
        "Created %s: rank=%s alpha=%s dropout=%s target_modules=%s exclude_modules=%s",
        lora_cls.__name__,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
        target_modules,
        exclude_modules,
    )
    return lora


def lora_adapter_name(args: Namespace) -> str:
    return getattr(args, "lora_adapter_name", None) or LORA_ADAPTER_NAME


def lora_runtime_dir(args: Namespace, step: int) -> Path:
    root = Path(getattr(args, "save", None) or tempfile.gettempdir())
    return root / "vime_lora_runtime" / f"step_{step}"


def save_lora_adapter_for_vllm(model: Sequence[torch.nn.Module], args: Namespace, step: int) -> str:
    """Export the current actor adapter in PEFT format for vLLM runtime loading."""
    from megatron.bridge import AutoBridge

    import vime_plugins.megatron_bridge  # noqa: F401
    from vime.utils import megatron_bridge_utils

    save_dir = lora_runtime_dir(args, step)
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_rank0 = rank == 0

    if is_rank0:
        save_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())

    bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
        AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    )
    lora_state_dict: dict[str, torch.Tensor] = {}
    with megatron_bridge_utils.patch_megatron_model(model):
        for hf_name, weight, megatron_name in bridge.export_adapter_weights(
            model,
            cpu=False,
            show_progress=False,
        ):
            try:
                if weight.is_cuda:
                    torch.cuda.synchronize(weight.device)
                lora_state_dict[hf_name] = weight.detach().clone().contiguous().cpu()
            except Exception:
                logger.exception(
                    "Failed to materialize LoRA tensor %s from %s shape=%s dtype=%s device=%s",
                    hf_name,
                    megatron_name,
                    tuple(weight.shape),
                    weight.dtype,
                    weight.device,
                )
                raise

    if is_rank0:
        torch.save(lora_state_dict, save_dir / "adapter_model.bin")
        config = build_peft_lora_config(args)
        with open(save_dir / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        (save_dir / "STABLE").touch()
        logger.info("Saved vLLM LoRA adapter %s with %s tensors", save_dir, len(lora_state_dict))

    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())
    return str(save_dir)


def build_peft_lora_config(args: Namespace) -> dict[str, Any]:
    target_modules = args.target_modules
    if not target_modules:
        target_modules_hf = list(_ALL_LINEAR_HF_MODULES)
    elif all(module in _HF_MODULE_NAMES for module in target_modules):
        target_modules_hf = list(target_modules)
    else:
        target_modules_hf = convert_target_modules_to_hf(target_modules)

    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": target_modules_hf,
    }
