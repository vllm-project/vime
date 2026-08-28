from __future__ import annotations

import logging
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerSendWeightsArgs
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLWeightTransferEngine,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateWeightFromSparseDistributed:
    """Send bit-exact Megatron-Bridge HF deltas through sparse HCCL.

    Version zero is the serving checkpoint. The first update captures an HF
    snapshot collectively; later updates compare integer views on CPU and send
    only changed BF16/FP16/FP32 elements. All rollout TP workers receive the
    same global-HF patch and let their model loader select the local shard.
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
        if quantization_config:
            raise NotImplementedError(
                "Sparse HCCL weight sync currently supports unquantized "
                "rollout weights only"
            )
        self.args = args
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self._snapshot: dict[str, torch.Tensor] = {}
        self._baseline_captured = False
        self._model_update_groups = None
        self._iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._is_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        )
        if self._is_src_rank:
            self._group_name = "vime-sparse-hccl"

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        del engine_gpu_offsets
        self.rollout_engines = list(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock
        if self._is_src_rank:
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                self.rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    def disconnect_rollout_engines(self) -> None:
        if self._is_src_rank and self._model_update_groups is not None:
            disconnect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                self._model_update_groups,
                self.rollout_engines,
            )
            self._model_update_groups = None

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    def _iter_hf_tensors(self):
        for chunk in self._iterator.get_hf_weight_chunks(
            self.weights_getter(), progress_desc="Sparse HCCL diff"
        ):
            if self._is_src_rank:
                yield from chunk

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            for name, tensor in self._iter_hf_tensors():
                self._snapshot[name] = tensor.detach().cpu().clone()
            self._baseline_captured = True
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                logger.info(
                    "[sparse HCCL] captured version-zero snapshot for %d tensors",
                    len(self._snapshot),
                )
            return

        self.weight_version += 1
        started = time.perf_counter()
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            # Runtime/HF sparse patches are not checkpoint-format reloads.
            ray.get(
                [
                    engine.start_weight_update.remote(is_checkpoint_format=False)
                    for engine in self.rollout_engines
                ]
            )
        dist.barrier(group=get_gloo_group())

        changed = total = wire = 0
        try:
            for name, tensor in self._iter_hf_tensors():
                patch, tensor_changed, tensor_total = self._make_patch(name, tensor)
                changed += tensor_changed
                total += tensor_total
                if patch is None:
                    continue
                wire += (
                    patch.indices.numel() * patch.indices.element_size()
                    + patch.values.numel() * patch.values.element_size()
                )
                self._send_patch(patch, list(tensor.shape))
            if self._is_src_rank:
                torch.npu.synchronize()
        finally:
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                try:
                    ray.get(
                        [
                            engine.finish_weight_update.remote()
                            for engine in self.rollout_engines
                        ]
                    )
                finally:
                    ray.get(
                        [
                            engine.continue_generation.remote()
                            for engine in self.rollout_engines
                        ]
                    )
            dist.barrier(group=get_gloo_group())

        device = torch.device("npu", torch.npu.current_device())
        counts = torch.tensor([changed, total, wire], dtype=torch.int64, device=device)
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        if dist.get_rank() == 0:
            elapsed = time.perf_counter() - started
            self.update_weight_metrics.update(
                {
                    "perf/update_weights_density": changed / max(total, 1),
                    "perf/update_weights_wire_bytes": wire,
                    "perf/update_weights_sparse_hccl_time": elapsed,
                }
            )
            logger.info(
                "[sparse HCCL v=%d] density=%.4f%% wire=%.2f MB elapsed=%.3fs",
                self.weight_version,
                100 * changed / max(total, 1),
                wire / 1e6,
                elapsed,
            )

    def _make_patch(
        self, name: str, current: torch.Tensor
    ) -> tuple[SparseWeightPatch | None, int, int]:
        new = current.detach().cpu().contiguous()
        old = self._snapshot.get(name)
        if old is None or old.shape != new.shape or old.dtype != new.dtype:
            raise RuntimeError(f"Sparse HCCL snapshot mismatch for {name}")
        integer_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}.get(new.element_size())
        if integer_dtype is None:
            raise NotImplementedError(f"Unsupported sparse HCCL dtype {new.dtype} for {name}")
        changed = new.view(integer_dtype).reshape(-1) != old.view(
            integer_dtype
        ).reshape(-1)
        indices = torch.nonzero(changed).flatten()
        total = new.numel()
        if indices.numel() == 0:
            return None, 0, total
        if new.numel() >= 2**31:
            raise OverflowError(f"Sparse HCCL int32 index limit exceeded for {name}: {new.numel()}")
        values = (
            new.reshape(-1)
            .index_select(0, indices)
            .to(device=current.device, non_blocking=False)
            .contiguous()
        )
        patch = SparseWeightPatch(
            name=name,
            indices=indices.to(device=current.device, dtype=torch.int32).contiguous(),
            values=values,
        )
        self._snapshot[name] = new.clone()
        return patch, patch.indices.numel(), total

    def _send_patch(self, patch: SparseWeightPatch, shape: list[int]) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = [
                engine.update_sparse_weights_from_distributed.remote(
                    names=[patch.name],
                    dtypes=[patch.values.dtype],
                    shapes=[shape],
                    num_updates_list=[patch.indices.numel()],
                    group_name=self._group_name,
                    weight_version=str(self.weight_version),
                )
                for engine in self.rollout_engines
            ]
            SparseHCCLWeightTransferEngine.trainer_send_weights(
                iter([patch]),
                HCCLTrainerSendWeightsArgs(group=self._model_update_groups, packed=False),
            )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
