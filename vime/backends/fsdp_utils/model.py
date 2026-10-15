"""Build, shard and load HF causal LMs with FSDP2 (``torch.distributed.fsdp.fully_shard``).

Model construction and the rank-0 load/broadcast flow are adapted from slime's
``FSDPTrainRayActor`` (https://github.com/THUDM/slime,
``slime/backends/fsdp_utils/actor.py`` at 0104a9e9, Apache-2.0), which in turn
references verl's ``fsdp_utils``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def resolve_param_dtype(args) -> torch.dtype:
    """Compute dtype of the sharded model; ``auto`` follows the shared ``--fp16`` flag."""
    name = args.fsdp_param_dtype
    if name == "auto":
        name = "fp16" if args.fp16 else "bf16"
    return _DTYPES[name]


def is_hf_checkpoint(path: str | None) -> bool:
    return path is not None and os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


def initial_weights_path(args) -> str:
    """HF directory the trainable weights start from when no FSDP checkpoint is resumed."""
    for path in (args.load, args.ref_load):
        if is_hf_checkpoint(path):
            return path
    return args.hf_checkpoint


def apply_fsdp2(model, mesh, args) -> torch.nn.Module:
    """Shard every decoder layer, then the root module (embeddings, final norm, LM head)."""
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    layer_cls_to_wrap = set(model._no_split_modules or [])
    assert layer_cls_to_wrap, f"{type(model).__name__} does not declare _no_split_modules"
    reduce_dtype = _DTYPES[args.fsdp_reduce_dtype]
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=resolve_param_dtype(args), reduce_dtype=reduce_dtype),
        "reshard_after_forward": args.fsdp_reshard_after_forward,
    }
    if args.fsdp_cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    tie = getattr(model.config, "tie_word_embeddings", False)
    for module in model.modules():
        # A tied embedding shares its weight with the root-owned LM head.
        if type(module).__name__ in layer_cls_to_wrap or (isinstance(module, torch.nn.Embedding) and not tie):
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    return model


def build_fsdp_model(args, hf_config, weights_path: str, mesh, device: torch.device):
    """Create the sharded trainable model with weights read once on rank 0.

    Rank 0 builds the model from ``weights_path``; the other ranks start from
    meta tensors and receive their shards through a rank-0 broadcast, so only
    one process reads the checkpoint. Non-persistent buffers (e.g. rotary
    frequencies) are not part of the state dict and are broadcast separately.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from transformers import AutoModelForCausalLM

    attn = args.fsdp_attn_implementation
    if dist.get_rank() == 0:
        model = AutoModelForCausalLM.from_pretrained(
            weights_path, dtype=torch.float32, attn_implementation=attn, trust_remote_code=True
        )
        full_state = model.state_dict()
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                hf_config, dtype=torch.float32, attn_implementation=attn, trust_remote_code=True
            )
        full_state = {}

    if args.fsdp_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    model = apply_fsdp2(model, mesh, args)
    if dist.get_rank() == 0:
        model.to(device)
    else:
        model.to_empty(device=device)

    cpu_offload = args.fsdp_cpu_offload
    set_model_state_dict(
        model,
        full_state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=cpu_offload),
    )
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)
    if cpu_offload:
        # CPUOffloadPolicy keeps sharded parameters on CPU; buffers stay on the device.
        model.to("cpu")
        for buffer in model.buffers():
            buffer.data = buffer.data.to(device)
    model.train()
    return model


def load_hf_weights_into(model, path: str, attn_implementation: str, cpu_offload: bool = False) -> None:
    """Overwrite the sharded model with an HF checkpoint read on rank 0 (in fp32, the master dtype)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from transformers import AutoModelForCausalLM

    full_state = {}
    if dist.get_rank() == 0:
        full_state = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation=attn_implementation, trust_remote_code=True
        ).state_dict()
    set_model_state_dict(
        model,
        full_state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=cpu_offload),
    )


def build_optimizer(args, model) -> torch.optim.Optimizer:
    """AdamW with Megatron's parameter grouping: biases and 1-D params get no weight decay."""
    if args.optimizer != "adam":
        raise ValueError(f"The fsdp2 backend supports --optimizer adam only, got {args.optimizer!r}")
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if name.endswith(".bias") or param.ndim == 1 else decay).append(param)
    groups = [
        {"params": decay, "wd_mult": 1.0, "lr_mult": 1.0},
        {"params": no_decay, "wd_mult": 0.0, "lr_mult": 1.0},
    ]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def local_shards(model) -> Iterator[tuple[str, torch.Tensor]]:
    """``(name, local shard)`` for every parameter; writing to a shard updates the parameter."""
    for name, param in model.named_parameters():
        yield name, param.to_local() if isinstance(param, DTensor) else param


class ShardBackup:
    """Host copies of this rank's parameter shards, one set per model tag.

    Used to keep several models (actor, reference) in one sharded module: a
    tag costs ``1 / world_size`` of the model in host memory per rank.
    """

    def __init__(self, model) -> None:
        self.model = model
        self._backups: dict[str, dict[str, torch.Tensor]] = {}
        self._pin = torch.cuda.is_available()

    @property
    def tags(self) -> list[str]:
        return list(self._backups)

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup = self._backups.setdefault(tag, {})
        for name, shard in local_shards(self.model):
            if name not in backup:
                backup[name] = torch.empty(shard.shape, dtype=shard.dtype, device="cpu", pin_memory=self._pin)
            backup[name].copy_(shard, non_blocking=True)
        _synchronize(self.model)

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup = self._backups[tag]
        for name, shard in local_shards(self.model):
            shard.copy_(backup[name], non_blocking=True)
        _synchronize(self.model)


def _synchronize(model) -> None:
    if torch.cuda.is_available() and any(p.is_cuda for p in model.parameters()):
        torch.cuda.synchronize()


@torch.no_grad()
def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move optimizer state tensors (e.g. Adam moments) between host and device.

    Adapted from slime's ``move_torch_optimizer`` (ref: verl ``fsdp_utils``).
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device, non_blocking=True)


def total_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
    """Clip gradients by their global L2 norm (``max_norm <= 0`` only measures) and return the norm."""
    norm = torch.nn.utils.clip_grad_norm_(list(parameters), max_norm if max_norm > 0 else float("inf"))
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    return float(norm)
