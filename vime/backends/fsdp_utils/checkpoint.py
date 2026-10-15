"""Distributed checkpoints for the FSDP2 backend.

Adapted from slime's ``slime/backends/fsdp_utils/checkpoint.py``
(https://github.com/THUDM/slime at 0104a9e9, Apache-2.0). The layout follows
vime's checkpoint contract so ``resolve_checkpoint`` and ``commit_checkpoint``
work unchanged::

    <save>/latest_checkpointed_iteration.txt   # holds the rollout id
    <save>/iter_<rollout_id:07d>/model/          # torch.distributed.checkpoint
    <save>/iter_<rollout_id:07d>/optimizer/      # unless --no-save-optim
    <save>/iter_<rollout_id:07d>/scheduler.pt    # unless --no-save-optim
    <save>/iter_<rollout_id:07d>/rng/rank_<r>.pt # unless --no-save-rng
    <save>/iter_<rollout_id:07d>/meta.json
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)

TRACKER = "latest_checkpointed_iteration.txt"
FORMAT = "vime-fsdp2"


class _ModelState(Stateful):
    def __init__(self, model) -> None:
        self.model = model

    def state_dict(self):
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict) -> None:
        set_model_state_dict(self.model, state_dict)


class _OptimizerState(Stateful):
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        return get_optimizer_state_dict(self.model, self.optimizer)

    def load_state_dict(self, state_dict) -> None:
        set_optimizer_state_dict(self.model, self.optimizer, state_dict)


def checkpoint_dir(root: str | Path, rollout_id: int) -> Path:
    return Path(root).expanduser() / f"iter_{rollout_id:07d}"


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def save(trainer, rollout_id: int) -> Path:
    """Save model shards, optimizer, schedule and RNG; publish the tracker last."""
    args = trainer.args
    root = Path(args.save).expanduser()
    directory = checkpoint_dir(root, rollout_id)
    if dist.get_rank() == 0:
        directory.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dcp.save({"model": _ModelState(trainer.model)}, checkpoint_id=str(directory / "model"))
    save_optim = not args.no_save_optim
    if save_optim:
        dcp.save(
            {"optimizer": _OptimizerState(trainer.model, trainer.optimizer)},
            checkpoint_id=str(directory / "optimizer"),
        )
    if not args.no_save_rng:
        rng_dir = directory / "rng"
        if dist.get_rank() == 0:
            rng_dir.mkdir(exist_ok=True)
        dist.barrier()
        torch.save(_rng_state(), rng_dir / f"rank_{dist.get_rank()}.pt")

    if dist.get_rank() == 0:
        if save_optim:
            torch.save(trainer.scheduler.state_dict(), directory / "scheduler.pt")
        meta = {
            "format": FORMAT,
            "rollout_id": rollout_id,
            "global_step": trainer.global_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    dist.barrier()
    if dist.get_rank() == 0:
        tracker = root / TRACKER
        tmp = tracker.with_suffix(".tmp")
        tmp.write_text(str(rollout_id))
        tmp.replace(tracker)
        logger.info(f"[FSDP] Saved checkpoint to {directory}")
    dist.barrier()
    return directory


def load(trainer) -> int | None:
    """Resume from ``--load`` when it holds an FSDP2 checkpoint; return the loaded rollout id.

    Returns ``None`` when ``--load`` has no tracker (a fresh start from HF weights).
    """
    args = trainer.args
    root = args.load
    if root is None or not (Path(root) / TRACKER).exists():
        return None
    step = args.ckpt_step if args.ckpt_step is not None else int((Path(root) / TRACKER).read_text())
    directory = checkpoint_dir(root, step)
    if not (directory / "model" / ".metadata").exists():
        raise ValueError(
            f"{directory} is not an fsdp2 checkpoint (no model/.metadata). Megatron checkpoints cannot be "
            "loaded by the fsdp2 backend; start from an HF checkpoint via --hf-checkpoint/--ref-load instead."
        )

    dcp.load({"model": _ModelState(trainer.model)}, checkpoint_id=str(directory / "model"))
    if not args.no_load_optim:
        if (directory / "optimizer" / ".metadata").exists():
            dcp.load(
                {"optimizer": _OptimizerState(trainer.model, trainer.optimizer)},
                checkpoint_id=str(directory / "optimizer"),
            )
        else:
            logger.warning(f"[FSDP] {directory} has no optimizer state; starting the optimizer fresh.")
        if (directory / "scheduler.pt").exists():
            trainer.scheduler.load_state_dict(torch.load(directory / "scheduler.pt", weights_only=True))
    rng_path = directory / "rng" / f"rank_{dist.get_rank()}.pt"
    if not args.no_load_rng and rng_path.exists():
        _set_rng_state(torch.load(rng_path, weights_only=False))

    meta_path = directory / "meta.json"
    if meta_path.exists():
        trainer.global_step = int(json.loads(meta_path.read_text()).get("global_step", 0))
    dist.barrier()
    logger.info(f"[FSDP] Loaded checkpoint {directory}")
    return step


def _hf_class_name(model) -> str:
    from torch.distributed.fsdp import FSDPModule

    return next(cls.__name__ for cls in type(model).__mro__ if not issubclass(cls, FSDPModule))


def save_hf(trainer, path: str | Path, tokenizer=None) -> None:
    """Write the trainable weights as an HF checkpoint (rank 0 gathers the full state dict)."""
    path = Path(path)
    state = get_model_state_dict(trainer.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if dist.get_rank() == 0:
        path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(path, state_dict=state)
        # fully_shard swaps in a generated subclass ("FSDP<Model>"); record the HF class.
        trainer.model.config.architectures = [_hf_class_name(trainer.model)]
        trainer.model.config.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
        logger.info(f"[FSDP] Saved HF checkpoint to {path}")
    dist.barrier()
