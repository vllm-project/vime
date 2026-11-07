import platform
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch

from vime.utils import accelerator

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper:
    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter
        # WSL limits pinned host memory for complete model snapshots.
        self._pin_memory = "microsoft" not in platform.release().lower()
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=self._pin_memory)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        accelerator.synchronize()

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        accelerator.synchronize()


@contextmanager
def tensors_from_backup(params: Iterable[tuple[str, torch.Tensor]], backup: dict[str, torch.Tensor]) -> Iterator[None]:
    """Let ``params`` read the values in ``backup`` for the duration of the block, then restore their storage.

    Assigning ``param.data`` swaps a parameter's storage. A distributed optimizer keeps writing its updates into
    the parameter's original DDP buffer view, so a parameter that is not pointed back at that view after the
    block keeps its old values for good.
    """
    original = []
    try:
        for name, param in params:
            original.append((param, param.data))
            param.data = backup[name].to(param.device)
        yield
    finally:
        for param, data in original:
            param.data = data
