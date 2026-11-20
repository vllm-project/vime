"""Sparse-coordinate probes backed by VIME's existing full HF converter."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseHFProbe:
    """Reconstruct only a local placeholder, never a distributed full weight.

    Missing TP shards contain NaN. The ordinary merge and model converter
    therefore carry real changed values to their exact checkpoint positions.
    This deliberately retains bounded dense probe buffers, not O(delta) memory.
    """

    args: object
    model_name: str
    info: object
    tp_rank: int
    tp_size: int

    def megatron_to_hf(self, tensor):
        from ..megatron_to_hf import convert_to_hf
        from ..update_weight.common import _merge_param_partitions

        attrs = self.info.attrs
        sharded = attrs.get("tensor_model_parallel", False) and attrs.get("parallel_mode") != "duplicated"
        if sharded and self.tp_size > 1:
            stride = attrs.get("partition_stride", 1)
            if stride != 1 and not (stride == 2 and "linear_fc1" in self.info.name):
                raise ValueError(f"Unsupported sparse partition stride {stride}: {self.info.name}")
            missing = torch.full_like(tensor, float("nan"))
            partitions = [missing] * self.tp_size
            partitions[self.tp_rank] = tensor
            tensor = _merge_param_partitions(
                self.info.name,
                partitions,
                attrs["partition_dim"],
                attrs.get("partition_sections"),
            )
        converted = convert_to_hf(self.args, self.model_name, self.info.name, tensor, quantization_config=None)
        result = {}
        for name, value in converted:
            if name in result or value.dtype != tensor.dtype:
                raise ValueError(f"Unsupported sparse conversion output: {name} (duplicate name or dtype change)")
            result[name] = value
        return result


class SparseHFExporter:
    """Share native VIME conversion between full seeds and sparse probes."""

    def __init__(self, args, model, *, model_name, quantization_config):
        if quantization_config is not None:
            raise ValueError("VIME sparse coordinate conversion requires unquantized floating weights")
        if getattr(args, "dspark_enabled", False) or getattr(args, "enable_mtp_training", False):
            raise ValueError("VIME sparse coordinate conversion does not support draft/MTP weights")
        from ..update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

        self.args = args
        self.model = model
        self.model_name = model_name
        self.direct = HfWeightIteratorDirect(args, model, model_name, quantization_config=None)
        self.param_infos = [info for bucket in self.direct.megatron_local_param_info_buckets for info in bucket]
        self.param_infos.sort(key=lambda info: info.name)

    def local_weights(self):
        from ..update_weight.common import named_params_and_buffers

        return dict(named_params_and_buffers(self.args, self.model))

    def get_hf_weight_chunks(self):
        # Export the live model, never a possibly stale training CPU backup.
        yield from self.direct.get_hf_weight_chunks(self.local_weights())
