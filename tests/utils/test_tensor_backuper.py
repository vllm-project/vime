"""Exporting the online DSpark draft from its CPU backup must not detach it from the optimizer's buffer.

``Actor.update_weights`` temporarily points the draft parameters at their CPU backup while the training memory
is paused. Megatron's distributed optimizer writes updates into each parameter's DDP buffer view, so a draft
parameter left pointing at the backup copy never changes again.
"""

import pytest
import torch

from vime.utils.tensor_backper import tensors_from_backup

NUM_GPUS = 0


def test_backup_values_are_visible_inside_and_buffer_views_come_back():
    buffer = torch.zeros(6)
    draft, other = torch.nn.Parameter(buffer[:3]), torch.nn.Parameter(buffer[3:])
    backup = {"draft": torch.tensor([1.0, 2.0, 3.0])}

    with tensors_from_backup([("draft", draft)], backup):
        assert draft.tolist() == [1.0, 2.0, 3.0]
        assert draft.data_ptr() != buffer.data_ptr()

    assert draft.data_ptr() == buffer.data_ptr()
    buffer.add_(5.0)  # an optimizer step writing into the buffer
    assert draft.tolist() == [5.0, 5.0, 5.0]
    assert other.tolist() == [5.0, 5.0, 5.0]


def test_buffer_views_come_back_when_the_export_fails():
    buffer = torch.zeros(2)
    draft = torch.nn.Parameter(buffer)
    with pytest.raises(RuntimeError, match="export failed"):
        with tensors_from_backup([("draft", draft)], {"draft": torch.ones(2)}):
            raise RuntimeError("export failed")
    assert draft.data_ptr() == buffer.data_ptr()
