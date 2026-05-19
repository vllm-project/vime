from __future__ import annotations

import logging
import os
import socket
import time
import traceback
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NcclBridge: isolate vLLM's PyNcclCommunicator in a subprocess so that it
# never coexists with torch.distributed NCCL groups in the Megatron trainer.
#
# vLLM's weight transfer uses raw NCCL (PyNcclCommunicator) which conflicts
# with torch.distributed's NCCL backend when both exist in the same process
# (see https://github.com/vllm-project/vllm/issues/5477). SGLang avoids
# this because it uses torch.distributed process groups for weight sync.
# ---------------------------------------------------------------------------


def _nccl_bridge_worker(
    conn,
    master_address: str,
    master_port: int,
    world_size: int,
    device: int,
    cvd: str,
    env_snapshot: dict[str, str],
) -> None:
    """Subprocess entry-point: creates PyNcclCommunicator and serves requests.

    GPU tensors are shared from the parent via CUDA IPC (torch.multiprocessing
    handles this transparently). No GPU→CPU→GPU copies are needed.

    Protocol over *conn* (multiprocessing.Connection):
    parent → child:
      {"op": "broadcast", "tensors": [gpu_tensor, ...]}
      {"op": "send_packed", "named_tensors": [(name, gpu_tensor), ...]}
      None → shutdown
    child → parent:
      "ready" (after init)
      "ok" (after each op)
      "error: ..."
    """
    try:
        os.environ.update(env_snapshot)
        if cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = cvd

        import torch as _torch  # noqa: PLC0415 — subprocess needs fresh import
        import torch.multiprocessing  # noqa: F401, PLC0415 — register CUDA IPC reducers

        _torch.cuda.set_device(device)

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: PLC0415
        from vllm.distributed.utils import StatelessProcessGroup  # noqa: PLC0415

        pg = StatelessProcessGroup.create(
            host=master_address,
            port=master_port,
            rank=0,
            world_size=world_size,
        )
        comm = PyNcclCommunicator(pg, device=device)

        conn.send("ready")

        while True:
            cmd = conn.recv()
            if cmd is None:
                break

            op = cmd["op"]
            if op == "broadcast":
                for t in cmd["tensors"]:
                    comm.broadcast(t, src=0, stream=_torch.cuda.current_stream())
                _torch.cuda.synchronize()
                conn.send("ok")

            elif op == "send_packed":
                # Prefer NCCLWeightTransferEngine.trainer_send_weights (newer vLLM). Some pip builds omit
                # NCCLTrainerSendWeightsArgs but still ship packed_broadcast_producer.
                try:
                    from vllm.distributed.weight_transfer.nccl_engine import (  # noqa: PLC0415
                        NCCLTrainerSendWeightsArgs,
                        NCCLWeightTransferEngine,
                    )

                    trainer_args = NCCLTrainerSendWeightsArgs(
                        group=comm,
                        packed=True,
                    )
                    NCCLWeightTransferEngine.trainer_send_weights(
                        iterator=iter(cmd["named_tensors"]),
                        trainer_args=trainer_args,
                    )
                except ImportError:
                    from vllm.distributed.weight_transfer.packed_tensor import (  # noqa: PLC0415
                        DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                        DEFAULT_PACKED_NUM_BUFFERS,
                        packed_broadcast_producer,
                    )

                    packed_broadcast_producer(
                        iterator=iter(cmd["named_tensors"]),
                        group=comm,
                        src=0,
                        post_iter_func=lambda x: x[1],
                        buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
                        num_buffers=DEFAULT_PACKED_NUM_BUFFERS,
                    )
                _torch.cuda.synchronize()
                conn.send("ok")

    except Exception as e:
        try:
            conn.send(f"error: {e}")
        except Exception:
            pass
        traceback.print_exc()


class _NcclBridge:
    """Runs vLLM's PyNcclCommunicator in a separate subprocess.

    This prevents NCCL communicator conflicts with torch.distributed groups
    that already exist in the Megatron trainer process. GPU tensors are shared
    with the subprocess via CUDA IPC (handled transparently by
    torch.multiprocessing), avoiding any GPU→CPU→GPU copies.
    """

    def __init__(self, master_address: str, master_port: int, world_size: int, device: int):
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()

        env_snapshot = dict(os.environ)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        self._process = ctx.Process(
            target=_nccl_bridge_worker,
            args=(child_conn, master_address, master_port, world_size, device, cvd, env_snapshot),
            daemon=True,
        )
        self._process.start()

        msg = self._parent_conn.recv()
        if isinstance(msg, str) and msg.startswith("error:"):
            raise RuntimeError(f"NcclBridge init failed: {msg}")
        if msg != "ready":
            raise RuntimeError(f"NcclBridge init unexpected response: {msg}")
        logger.info("NcclBridge ready (pid=%d, device=%d)", self._process.pid, device)

    def broadcast_tensors(self, tensors: list[torch.Tensor]) -> None:
        """Broadcast a list of tensors (one-by-one) via the bridge subprocess."""
        gpu_tensors = [t.contiguous() for t in tensors]
        self._parent_conn.send({"op": "broadcast", "tensors": gpu_tensors})
        self._wait_ok("broadcast_tensors")

    def send_weights_packed(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Send weights using vLLM's packed broadcast protocol."""
        gpu_pairs = []
        for name, t in named_tensors:
            data = t.data if hasattr(t, "data") else t
            gpu_pairs.append((name, data.contiguous()))
        self._parent_conn.send({"op": "send_packed", "named_tensors": gpu_pairs})
        self._wait_ok("send_weights_packed")

    def _wait_ok(self, label: str, timeout: float = 600.0) -> None:
        if not self._parent_conn.poll(timeout):
            raise TimeoutError(f"NcclBridge {label} timed out after {timeout}s")
        msg = self._parent_conn.recv()
        if msg != "ok":
            raise RuntimeError(f"NcclBridge {label} failed: {msg}")

    def shutdown(self) -> None:
        try:
            self._parent_conn.send(None)
            self._process.join(timeout=30)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or self._model_update_groups is None:
            return
        disconnect_rollout_engines_from_distributed(
            self.args, self._group_name, self._model_update_groups, self.rollout_engines
        )
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        use_vllm_packed = self._use_vllm_packed()
        if use_vllm_packed and self._is_pp_src_rank:
            logger.info(
                "Using vLLM packed weight sync (bucketed; metadata + trainer_send_weights per bucket)"
            )

        if use_vllm_packed:
            buffer_size = 0
            converted_named_tensors: list[tuple[str, torch.Tensor]] = []
            pbar = (
                tqdm(desc=f"[{self._group_name}] Update weights (vLLM packed)", total=0)
                if self._is_pp_src_rank
                else None
            )
            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." in name:
                    continue
                buffer_size = self._update_weight_from_distributed(
                    name,
                    param,
                    converted_named_tensors,
                    buffer_size,
                    pbar=pbar,
                    flush_packed=True,
                )
            if converted_named_tensors and self._is_pp_src_rank:
                self._update_weights_vllm_packed(converted_named_tensors)
                if pbar is not None:
                    pbar.update(1)
        else:
            buffer_size = 0
            converted_named_tensors = []
            pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." in name:
                    continue
                buffer_size = self._update_weight_from_distributed(
                    name, param, converted_named_tensors, buffer_size, pbar=pbar
                )

            if converted_named_tensors:
                self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        if not use_vllm_packed:
            buffer_size = 0
            named_tensors = []
            pbar = tqdm(desc=f"[{self._group_name}] Update weights (experts)", total=0) if self._is_pp_src_rank else None
            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." not in name:
                    continue
                buffer_size = self._update_expert_weight_from_distributed(
                    name, param, named_tensors, buffer_size, pbar=pbar
                )

            if named_tensors:
                self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _use_vllm_packed(self) -> bool:
        """Use vLLM packed weight transfer (one-shot metadata + trainer_send_weights)."""
        if not getattr(self.args, "vllm_weight_sync_packed", True):
            return False
        if any(".experts." in name for name, _ in named_params_and_buffers(self.args, self.model)):
            return False
        if self.quantization_config and self.quantization_config.get("quant_method") == "compressed-tensors":
            return False
        return True

    def _update_weights_vllm_packed(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Single-shot vLLM weight update using packed broadcast."""
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                packed=True,
            )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
        *,
        flush_packed: bool = False,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            if converted_named_tensors:
                if flush_packed:
                    self._update_weights_vllm_packed(converted_named_tensors)
                    converted_named_tensors.clear()
                    if pbar is not None:
                        pbar.update(1)
                else:
                    self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
            packed=False,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        if pbar is not None:
            pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> Any:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.

    For vLLM backend, the trainer-side NCCL communicator is created inside a
    separate subprocess (_NcclBridge) to avoid conflicts between vLLM's raw
    NCCL (PyNcclCommunicator) and the torch.distributed NCCL groups that
    Megatron already holds in this process.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    device = torch.cuda.current_device()
    logger.info(
        "vLLM weight transfer via NcclBridge: addr=%s port=%d world_size=%d device=%d CVD=%s",
        master_address,
        master_port,
        world_size,
        device,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    model_update_groups = _NcclBridge(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
        device=device,
    )

    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    model_update_groups: Any,
    rollout_engines: Sequence[ActorHandle],
) -> None:
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    if isinstance(model_update_groups, _NcclBridge):
        model_update_groups.shutdown()
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: "_NcclBridge",
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    packed: bool = False,
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → vLLM engines via
    the ``_NcclBridge`` subprocess so that raw NCCL never runs inside the
    Megatron trainer process).
    """
    kwargs: dict[str, Any] = {
        "names": [name for name, _ in converted_named_tensors],
        "dtypes": [param.dtype for _, param in converted_named_tensors],
        "shapes": [param.shape for _, param in converted_named_tensors],
        "group_name": group_name,
        "weight_version": str(weight_version),
        "packed": packed,
    }

    refs = [engine.update_weights_from_distributed.remote(**kwargs) for engine in rollout_engines]

    if packed:
        group.send_weights_packed(list(converted_named_tensors))
    else:
        group.broadcast_tensors([param.data for _, param in converted_named_tensors])

    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
