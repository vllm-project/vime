import torch

from .common import SafetensorReader, strip_mcore_wrappers


def ouro_hf_tensor(name: str, reader: SafetensorReader, config: object) -> torch.Tensor:
    # The shared-layer provider retains the checkpoint's physical parameter layout.
    physical_name = strip_mcore_wrappers(name)
    if physical_name == "output_layer.weight":
        physical_name = "lm_head.weight"
    return reader.get_tensor(physical_name)
