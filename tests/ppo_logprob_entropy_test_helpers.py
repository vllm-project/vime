"""Shared CUDA parity and NPU mathematical-reference input scenarios."""

import torch

PARITY_SCENARIOS = [
    (-1, False, False, False),
    (-1, False, True, False),
    (-1, False, True, True),
    (-1, True, False, False),
    (-1, True, True, False),
    (-1, True, True, True),
    (2, False, False, False),
    (2, False, True, False),
    (2, False, True, True),
    (2, True, False, False),
    (2, True, True, False),
    (2, True, True, True),
]


def _full_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, -1.0],
            [4.0, 1.0, 0.5, 2.0, 3.0, 0.0],
            [-1.0, 3.0, 2.0, 0.0, 1.0, 5.0],
            [0.2, -0.4, 1.7, -2.0, 3.3, 0.0],
        ],
        dtype=torch.float32,
    )


def _keep_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [False, True, False, True, False, False],  # target 5 is absent.
            [False, True, False, False, True, False],  # target 0 is absent.
            [True, False, True, False, False, True],  # target 3 is absent.
            [False, True, False, True, False, False],  # target 2 is absent.
        ],
        dtype=torch.bool,
    )
