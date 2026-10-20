"""Small, backend-neutral accelerator contract used by Vime.

The contract intentionally contains only operations that Vime uses in its
runtime.  Vendor modules are supplied by concrete implementations and are
never imported by this module.
"""

from __future__ import annotations

import abc
import os
from collections.abc import Mapping
from typing import Any

import torch


class Accelerator(abc.ABC):
    """Common device/runtime surface exposed to Vime code."""

    name: str
    device_type: str
    communication_backend_name: str

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return whether this backend can actually execute on this host."""

    @abc.abstractmethod
    def device(self, index: int | str | torch.device | None = None) -> torch.device:
        """Return a :class:`torch.device` for a local device index."""

    @abc.abstractmethod
    def device_name(self, index: int | str | torch.device | None = None) -> str:
        """Return the canonical device string used by PyTorch APIs."""

    @abc.abstractmethod
    def set_device(self, index: int | str | torch.device) -> None:
        """Select the current local device."""

    @abc.abstractmethod
    def current_device(self) -> int | str:
        """Return the current local device index, or ``cpu`` for CPU."""

    @abc.abstractmethod
    def device_count(self) -> int:
        """Return the number of visible devices."""

    @abc.abstractmethod
    def synchronize(self, device: int | str | torch.device | None = None) -> None:
        """Synchronize work on one device or the current device."""

    @abc.abstractmethod
    def current_stream(self, device: int | str | torch.device | None = None) -> Any:
        """Return the current stream, or ``None`` when streams are unsupported."""

    def default_stream(self, device: int | str | torch.device | None = None) -> Any:
        """Return the default stream, or ``None`` when streams are unsupported."""
        return None

    def stream(self, stream: Any):
        """Return a context manager for a stream when the backend supports it."""
        raise NotImplementedError(f"Accelerator {self.name!r} does not support streams")

    @property
    def Stream(self) -> Any:
        return None

    @property
    def Event(self) -> Any:
        return None

    @abc.abstractmethod
    def empty_cache(self) -> None:
        """Release allocator-held, currently unused memory."""

    def ipc_collect(self) -> None:
        """Collect inter-process allocator state when supported."""
        return None

    def set_allocator_expandable_segments(self) -> bool:
        """Configure expandable allocator segments when supported."""
        return False

    @abc.abstractmethod
    def mem_get_info(self, device: int | str | torch.device | None = None) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)`` for the selected device."""

    @abc.abstractmethod
    def memory_allocated(self, device: int | str | torch.device | None = None) -> int:
        """Return currently allocated device memory in bytes."""

    @abc.abstractmethod
    def memory_reserved(self, device: int | str | torch.device | None = None) -> int:
        """Return allocator-reserved device memory in bytes."""

    def get_device_properties(self, device: int | str | torch.device | None = None) -> Any:
        return None

    def device_uuid(self) -> str:
        """Return a stable identifier for the current device.

        Weight transfer names the local device in its handshake, so sender and
        receiver have to derive the same identifier the same way.
        """
        uuid = getattr(self.get_device_properties(self.current_device()), "uuid", None)
        if uuid is None:
            raise RuntimeError(f"Accelerator {self.name!r} does not expose a device UUID")
        return str(uuid)

    def memory_module(self) -> Any:
        """Return the backend memory namespace, if it exposes one."""
        return None

    def attach_oom_observer(self, callback) -> bool:
        """Attach an OOM callback; return ``False`` when unsupported."""
        return False

    def supports(self, capability: str) -> bool:
        """Return whether a named optional capability is implemented."""
        return False

    def autocast(self, *args, **kwargs):
        """Return an autocast context for this backend."""
        raise NotImplementedError(f"Accelerator {self.name!r} does not support autocast")

    def manual_seed(self, seed: int) -> None:
        """Seed the current device generator when supported."""
        return None

    def manual_seed_all(self, seed: int) -> None:
        """Seed all device generators when supported."""
        return None

    def get_rng_state(self, device: int | str | torch.device | None = None) -> torch.Tensor:
        """Return the current generator state."""
        return torch.get_rng_state()

    def set_rng_state(self, state: torch.Tensor, device: int | str | torch.device | None = None) -> None:
        """Restore the current generator state."""
        torch.set_rng_state(state)

    def initial_seed(self) -> int:
        return int(torch.initial_seed())

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> torch.device | None:
        """Return the device id accepted by ``dist.init_process_group``."""
        return self.device(index)

    def post_import_torch(self) -> None:
        """Apply an optional backend hook after third-party torch imports."""
        return None

    def communication_backend(self, default: str = "nccl") -> str:
        """Map a logical default backend to this accelerator's transport."""
        return self.communication_backend_name if default == "nccl" else default

    def weight_update_backend(self, default: str = "nccl") -> str:
        return self.communication_backend(default)

    # The two weight-transfer lookups below are classmethods on purpose: they
    # answer what a transport is called and which class implements it, which
    # needs no device. A trainer that only has to recognize a peer's backend
    # name, and a CPU-only tool, must not be made to select a runtime for it.

    @classmethod
    def weight_transfer_backend(cls, transport: str) -> str:
        """Map a logical weight transport onto the serving backend's name for it.

        ``ipc`` is a colocated handoff between processes sharing one device;
        ``distributed`` is a collective between separate trainer and engine
        groups. A backend that registers different names overrides this.
        """
        names = {"ipc": "ipc", "distributed": "nccl"}
        if transport not in names:
            raise ValueError(f"Unknown weight transport: {transport!r}")
        return names[transport]

    # Child-process environment. A training actor, a serving engine actor and
    # the serving subprocess that actor launches are started by three
    # different subsystems, but what each has to be told about the
    # accelerator is one question, so it is answered in one place. The first
    # two are built on the driver, which may own no device, so these are
    # classmethods like the lookups above.

    @classmethod
    def train_actor_env(cls, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment a training actor needs, including the offload allocator."""
        env = dict(env_vars or {})
        if not getattr(args, "offload_train", False):
            return env
        return cls.offload_allocator_env(env)

    @classmethod
    def engine_actor_env(cls, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment a serving engine actor needs."""
        return dict(env_vars or {})

    @classmethod
    def serving_subprocess_env(
        cls, base_env: Mapping[str, str], *, visible_devices: str, colocate: bool
    ) -> dict[str, str]:
        """Environment the serving subprocess needs, given its device selection."""
        return dict(base_env)

    @classmethod
    def offload_allocator_env(cls, env_vars: Mapping[str, str]) -> dict[str, str]:
        """Hand the allocator to torch_memory_saver in the actor process.

        All three variables are required for it to take over, so a missing
        library raises rather than letting training start without offload.
        """
        import torch_memory_saver

        for name in (
            "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
            "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
            "torch_memory_saver_hook_mode_preload.abi3.so",
        ):
            dynlib_path = os.path.join(os.path.dirname(os.path.dirname(torch_memory_saver.__file__)), name)
            if os.path.exists(dynlib_path):
                break
        else:
            raise FileNotFoundError(
                "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
            )

        env = dict(env_vars)
        env["LD_PRELOAD"] = dynlib_path
        env["TMS_INIT_ENABLE"] = "1"
        env["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
        return env

    @classmethod
    def megatron_lifecycle_qualname(cls) -> str | None:
        """Name a module replacing vime.backends.megatron_lifecycle, or None.

        A backend whose vendor patches must be installed at an exact point in
        Megatron's startup, or whose memory saver behaves differently, names
        its module here. None means the shared default, and nothing is
        imported.
        """
        return None

    @classmethod
    def weight_transfer_init_info_qualname(cls, transport: str) -> str:
        """Name the trainer init-info class for ``transport``.

        Returned as a qualified name rather than a class so that describing a
        transport never imports a serving runtime, and so a backend can name
        classes in its own vendor package without this module knowing it.
        """
        names = {
            "ipc": "vllm.distributed.weight_transfer.ipc_engine.IPCTrainerInitInfo",
            "distributed": "vllm.distributed.weight_transfer.nccl_engine.NCCLTrainerInitInfo",
        }
        if transport not in names:
            raise ValueError(f"Unknown weight transport: {transport!r}")
        return names[transport]

    ray_device_key = "GPU"
    """The Ray resource name this accelerator is advertised and scheduled under.

    Ray resolves a worker's visible-devices variable from the resource name it
    was scheduled against, so a backend addressed under the wrong name starts
    with no visible device. Backends whose devices Ray models separately from
    its own GPU resource override this together with the flag below.
    """

    ray_uses_native_gpu_resource = True
    """Whether ``ray_device_key`` is Ray's own GPU resource rather than a custom one."""

    @property
    def visible_devices_env(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    def resolve_visible_device_id(self, physical_device_id: int | float | str) -> int:
        """Map a physical id to a local id under this backend's visibility env."""
        raw_value = str(physical_device_id).strip()
        visible = os.environ.get(self.visible_devices_env)
        if not visible:
            return int(float(raw_value))

        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if raw_value in ids:
            return ids.index(raw_value)

        try:
            value = int(float(raw_value))
        except ValueError:
            value = None
        if value is not None and str(value) in ids:
            return ids.index(str(value))
        if value is not None and 0 <= value < len(ids):
            return value
        raise RuntimeError(
            f"Device id {raw_value} is not valid under {self.visible_devices_env}={visible}. "
            f"Expected one of {ids} (physical) or 0..{len(ids) - 1} (local)."
        )

    def accelerator_module(self) -> Any:
        """Return the torch backend namespace, or ``None`` for CPU."""
        return None
