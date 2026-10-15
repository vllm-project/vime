"""Training backends selectable with ``--train-backend``.

Each backend names its Ray training actor and, optionally, a function that
adds flags only that backend understands. Entries are ``"module:attribute"``
strings, so listing the choices or parsing the flags never imports a training
stack; the actor module is imported only when a training group is created.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainBackend:
    actor: str
    arguments: str | None = None


_TRAIN_BACKENDS: dict[str, TrainBackend] = {
    "megatron": TrainBackend(actor="vime.backends.megatron_utils.actor:MegatronTrainRayActor"),
    "fsdp2": TrainBackend(
        actor="vime.backends.fsdp_utils.actor:FSDPTrainRayActor",
        arguments="vime.backends.fsdp_utils.arguments:add_fsdp_arguments",
    ),
}


def train_backend_names() -> list[str]:
    return list(_TRAIN_BACKENDS)


def _get(name: str) -> TrainBackend:
    try:
        return _TRAIN_BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown training backend {name!r}; registered: {train_backend_names()}") from None


def _load(path: str):
    module_name, _, attribute = path.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def get_train_actor_cls(name: str) -> type:
    return _load(_get(name).actor)


def parse_train_backend_args(name: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the selected backend's own flags and leave every other flag alone.

    The Megatron parser ignores unknown flags, so backend-only flags would be
    silently dropped there; they are parsed here instead and merged into the
    final namespace together with ``--train-backend``.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    backend = _get(name)
    if backend.arguments is not None:
        parser = _load(backend.arguments)(parser)
    backend_args, _ = parser.parse_known_args(argv)
    return backend_args
