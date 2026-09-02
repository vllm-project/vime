import torch

from vime.utils.tensor_backper import TensorBackuper


def test_double_buffer_keeps_previous_backup_immutable(monkeypatch):
    source = torch.tensor([1.0, 2.0])
    original_empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda tensor, **kwargs: original_empty_like(tensor, device="cpu"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    backuper = TensorBackuper.create(
        source_getter=lambda: [("weight", source)], single_tag=None
    )
    backuper.enable_double_buffer("actor")

    backuper.backup("actor")
    first = backuper.get("actor")["weight"]
    source.add_(10)
    backuper.backup("actor")
    second = backuper.get("actor")["weight"]

    assert first.tolist() == [1.0, 2.0]
    assert second.tolist() == [11.0, 12.0]
    assert first.data_ptr() != second.data_ptr()

    source.add_(10)
    backuper.backup("actor")
    assert second.tolist() == [11.0, 12.0]
    assert backuper.get("actor")["weight"].tolist() == [21.0, 22.0]
