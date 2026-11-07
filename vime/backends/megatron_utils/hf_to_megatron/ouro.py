import torch

from .common import SafetensorReader, strip_mcore_wrappers


def ouro_hf_tensor(name: str, reader: SafetensorReader, config: object) -> torch.Tensor:
    # The shared-layer provider retains the checkpoint's physical parameter layout.
    return reader.get_tensor(strip_mcore_wrappers(name))
