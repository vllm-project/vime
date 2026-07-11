from __future__ import annotations

import ctypes
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

_STORE_CACHE: dict[tuple[tuple[str, str], ...], Any] = {}
_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def default_store_init_kwargs() -> dict[str, Any]:
    return {
        "local_hostname": os.getenv("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
        "metadata_server": os.getenv("MOONCAKE_TE_META_DATA_SERVER", "http://127.0.0.1:18080/metadata"),
        "global_segment_size": int(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(512 * 1024 * 1024))),
        "local_buffer_size": int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(128 * 1024 * 1024))),
        "protocol": os.getenv("MOONCAKE_PROTOCOL", "tcp"),
        "rdma_devices": os.getenv("MOONCAKE_DEVICE", ""),
        "master_server_addr": os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051"),
    }


def normalize_store_init_kwargs(store_init_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if not store_init_kwargs:
        return default_store_init_kwargs()
    return default_store_init_kwargs() | dict(store_init_kwargs)


def create_mooncake_store(store_init_kwargs: dict[str, Any] | None = None) -> Any:
    from mooncake.store import MooncakeDistributedStore  # type: ignore

    kwargs = normalize_store_init_kwargs(store_init_kwargs)
    store = MooncakeDistributedStore()
    if store.setup(**kwargs) != 0:
        raise RuntimeError("Mooncake Store setup failed")
    return store


def get_cached_mooncake_store(store_init_kwargs: dict[str, Any] | None = None) -> Any:
    kwargs = normalize_store_init_kwargs(store_init_kwargs)
    cache_key = tuple(sorted((key, repr(val)) for key, val in kwargs.items()))
    if cache_key not in _STORE_CACHE:
        _STORE_CACHE[cache_key] = create_mooncake_store(kwargs)
    return _STORE_CACHE[cache_key]


def remove_mooncake_keys(store: Any, keys: list[str]) -> None:
    errors = []
    for key in sorted(set(keys)):
        ret = store.remove(key, True)
        if ret != 0:
            errors.append((key, ret))
    if errors:
        raise RuntimeError(f"Mooncake key cleanup failed: {errors}")


@dataclass
class MooncakeRemoteBatch:
    """Remote tensor batch stored in Mooncake Store."""

    fields: dict[str, tuple[str, tuple[int, ...], str]]  # name -> (store_key, shape, dtype)
    batch_size: int
    store_init_kwargs: dict[str, Any] = field(default_factory=dict)
    keys_to_cleanup: tuple[str, ...] = ()

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        store: Any,
        prefix: str,
        store_init_kwargs: dict[str, Any] | None = None,
    ) -> MooncakeRemoteBatch:
        if not prefix or ".." in prefix:
            raise ValueError(f"invalid Mooncake key prefix: {prefix!r}")
        config = _hard_pin_config(store)
        fields: dict[str, tuple[str, tuple[int, ...], str]] = {}
        written_keys: list[str] = []
        batch_size = None
        try:
            for name, tensor in tensors.items():
                if _FIELD_NAME_RE.fullmatch(name) is None:
                    raise ValueError(f"invalid Mooncake tensor field name: {name!r}")
                cpu_tensor = tensor.detach().contiguous().cpu()
                if batch_size is None:
                    batch_size = int(cpu_tensor.shape[0])
                elif int(cpu_tensor.shape[0]) != batch_size:
                    raise ValueError(f"tensor {name} batch size mismatch")
                key = f"{prefix}/{name}"
                if _put_tensor(store, key, cpu_tensor, config) != 0:
                    raise RuntimeError(f"Mooncake put failed for {key}")
                written_keys.append(key)
                fields[name] = (key, tuple(cpu_tensor.shape), str(cpu_tensor.dtype).removeprefix("torch."))
        except Exception:
            remove_mooncake_keys(store, written_keys)
            raise
        return cls(
            fields=fields,
            batch_size=batch_size or 0,
            store_init_kwargs=store_init_kwargs or {},
            keys_to_cleanup=tuple(written_keys),
        )

    def __len__(self) -> int:
        return self.batch_size

    def materialize(self, fields: list[str] | None = None) -> dict[str, torch.Tensor]:
        store = get_cached_mooncake_store(self.store_init_kwargs)
        selected = list(self.fields) if fields is None else fields
        return {name: _get_tensor(store, *self.fields[name]) for name in selected}

    def cleanup(self) -> None:
        if self.keys_to_cleanup:
            remove_mooncake_keys(get_cached_mooncake_store(self.store_init_kwargs), list(self.keys_to_cleanup))


def _put_tensor(store: Any, key: str, tensor: torch.Tensor, config: Any) -> int:
    arr = np.ascontiguousarray(tensor.numpy())
    region = _WritableRegion(memoryview(arr))
    try:
        if store.register_buffer(region.ptr, region.size) != 0:
            raise RuntimeError("register_buffer failed for put_from")
        try:
            return store.put_from(key=key, buffer_ptr=region.ptr, size=region.size, config=config)
        finally:
            if store.unregister_buffer(region.ptr) != 0:
                raise RuntimeError("unregister_buffer failed for put_from")
    finally:
        region.close()


def _get_tensor(store: Any, key: str, shape: tuple[int, ...], dtype_name: str) -> torch.Tensor:
    torch_dtype = getattr(torch, dtype_name.lower())
    nbytes = int(np.prod(shape, dtype=np.int64)) * torch_dtype.itemsize
    if nbytes == 0:
        return torch.empty(shape, dtype=torch_dtype)
    region = _WritableRegion(bytearray(nbytes))
    try:
        if store.register_buffer(region.ptr, region.size) != 0:
            raise RuntimeError("register_buffer failed for get_into")
        try:
            got = store.get_into(key, region.ptr, nbytes)
            if got != nbytes:
                raise RuntimeError(f"get_into failed for {key}: expected {nbytes}, got {got}")
        finally:
            if store.unregister_buffer(region.ptr) != 0:
                raise RuntimeError("unregister_buffer failed for get_into")
        count = int(np.prod(shape, dtype=np.int64))
        return torch.frombuffer(region.buffer, dtype=torch_dtype, count=count).reshape(shape).clone()
    finally:
        region.close()


class _WritableRegion:
    def __init__(self, buffer: Any) -> None:
        self.buffer = buffer
        self.view = memoryview(buffer).cast("B")
        self.c_buffer = (ctypes.c_ubyte * self.view.nbytes).from_buffer(self.view)
        self.ptr = ctypes.addressof(self.c_buffer)
        self.size = self.view.nbytes

    def close(self) -> None:
        self.c_buffer = None
        self.view.release()


def _hard_pin_config(store: Any) -> Any:
    from mooncake.store import ReplicateConfig  # type: ignore

    config = ReplicateConfig()
    config.preferred_segments = [store.get_hostname()]
    config.with_hard_pin = True
    return config
