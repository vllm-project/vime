from .dspark import collate_dspark_samples, compute_dspark_loss, dspark_trainer_kwargs, sync_dspark_lm_heads
from .eagle3 import Eagle3AlignedSample, align_eagle3_sample, collate_eagle3_samples, compute_eagle3_loss

__all__ = [
    "Eagle3AlignedSample",
    "align_eagle3_sample",
    "collate_dspark_samples",
    "collate_eagle3_samples",
    "compute_dspark_loss",
    "compute_eagle3_loss",
    "dspark_trainer_kwargs",
    "sync_dspark_lm_heads",
]
