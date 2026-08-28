import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).parents[5]
    / "vime/backends/megatron_utils/update_weight/megatron_sparse_export.py"
)
SPEC = importlib.util.spec_from_file_location("test_megatron_sparse_export_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
export = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = export
SPEC.loader.exec_module(export)


def test_local_bit_exact_diff_uses_storage_bits() -> None:
    old_bits = torch.tensor([0x0000, 0x8000, 0x7FC1, 0x3F80], dtype=torch.uint16)
    new_bits = torch.tensor([0x8000, 0x8000, 0x7FC2, 0x3F80], dtype=torch.uint16)
    old = old_bits.view(torch.bfloat16)
    new = new_bits.view(torch.bfloat16)

    indices, values = export.local_bit_exact_diff(new, old)

    assert indices.tolist() == [0, 2]
    assert values.view(torch.uint16).tolist() == [0x8000, 0x7FC2]


class _SplitProbe:
    def megatron_to_hf(self, tensor, _module):
        # Mimic a fused Megatron parameter split into two final HF tensors.
        return {"q.weight": tensor[:2], "k.weight": tensor[2:]}


def test_sparse_hf_entry_splits_fused_parameter() -> None:
    record = export.SparseExportRecord(
        megatron_name="decoder.qkv.weight",
        weight_key="vp_stages.0.decoder.qkv.weight",
        param=torch.empty(4, dtype=torch.bfloat16),
        gather_group=None,
        contributes=True,
        probe=_SplitProbe(),
        module=object(),
    )
    cache = {}

    slots, counts, indices, values = export.sparse_hf_entry(
        record,
        torch.tensor([1, 3]),
        torch.tensor([2.0, 4.0], dtype=torch.bfloat16),
        cache,
    )

    assert slots == [("q.weight", (2,)), ("k.weight", (2,))]
    assert counts.tolist() == [1, 1]
    assert indices.tolist() == [1, 1]
    assert values.tolist() == [2.0, 4.0]


class _GlobalOffsetProbe:
    def megatron_to_hf(self, tensor, _module):
        missing = torch.full_like(tensor, float("nan"))
        return {"proj.weight": torch.cat((missing, tensor))}


def test_sparse_hf_entry_preserves_final_hf_global_indices() -> None:
    record = export.SparseExportRecord(
        megatron_name="decoder.proj.weight",
        weight_key="vp_stages.0.decoder.proj.weight",
        param=torch.empty(3, dtype=torch.bfloat16),
        gather_group=object(),
        contributes=True,
        probe=_GlobalOffsetProbe(),
        module=object(),
    )

    slots, counts, indices, values = export.sparse_hf_entry(
        record,
        torch.tensor([0, 2]),
        torch.tensor([5.0, 7.0], dtype=torch.bfloat16),
        {},
    )

    assert slots == [("proj.weight", (6,))]
    assert counts.tolist() == [2]
    assert indices.tolist() == [3, 5]
    assert values.tolist() == [5.0, 7.0]


def test_exchange_slot_tables_uses_gloo_control_group(monkeypatch) -> None:
    gloo_group = object()
    seen = {}
    record = SimpleNamespace(megatron_name="weight", slots=None)
    cache = {"weight": [("model.weight", (2, 2))]}

    monkeypatch.setattr(export, "get_gloo_group", lambda: gloo_group)
    monkeypatch.setattr(export.dist, "get_world_size", lambda: 1)

    def fake_all_gather_object(output, value, group=None):
        seen["group"] = group
        output[0] = value

    monkeypatch.setattr(export.dist, "all_gather_object", fake_all_gather_object)

    export._exchange_slot_tables([record], cache)

    assert seen["group"] is gloo_group
    assert record.slots == [("model.weight", (2, 2))]
