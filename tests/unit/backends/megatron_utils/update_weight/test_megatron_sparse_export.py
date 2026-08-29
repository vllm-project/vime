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


def test_clone_cpu_snapshot_does_not_alias_mutable_backup() -> None:
    backup = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    snapshot = export.clone_cpu_snapshot(backup)
    backup.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))

    indices, values = export.local_bit_exact_diff(backup, snapshot)
    assert indices.tolist() == [0, 1]
    assert values.tolist() == [3.0, 4.0]


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


def _mapping(class_name, **attributes):
    mapping = type(class_name, (), {})()
    for name, value in attributes.items():
        setattr(mapping, name, value)
    return mapping


def _fast_record(mapping, shape):
    return SimpleNamespace(
        mapping=mapping,
        param=torch.empty(shape, dtype=torch.bfloat16),
        module=SimpleNamespace(),
        megatron_name="weight",
        slots=None,
    )


def test_fast_column_and_row_coordinate_mapping() -> None:
    column = _fast_record(
        _mapping("ColumnParallelMapping", hf_param="column", tp_rank=2),
        (2, 3),
    )
    column.slots = [("column", (8, 3))]
    _slots, counts, indices, values = export.sparse_hf_entry(
        column,
        torch.tensor([0, 5]),
        torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        {},
    )
    assert counts.tolist() == [2]
    assert indices.tolist() == [12, 17]
    assert values.tolist() == [1.0, 2.0]

    row = _fast_record(
        _mapping("RowParallelMapping", hf_param="row", tp_rank=1),
        (2, 3),
    )
    row.slots = [("row", (2, 12))]
    _slots, counts, indices, _values = export.sparse_hf_entry(
        row,
        torch.tensor([0, 5]),
        torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        {},
    )
    assert counts.tolist() == [2]
    assert indices.tolist() == [3, 17]


def test_fast_gated_mlp_coordinate_mapping() -> None:
    mapping = _mapping(
        "GatedMLPMapping",
        hf_param={"gate": "gate", "up": "up"},
        tp_rank=1,
    )
    record = _fast_record(mapping, (4, 3))
    record.slots = [("gate", (4, 3)), ("up", (4, 3))]

    _slots, counts, indices, values = export.sparse_hf_entry(
        record,
        torch.tensor([0, 7, 11]),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
        {},
    )

    assert counts.tolist() == [1, 2]
    assert indices.tolist() == [6, 7, 11]
    assert values.tolist() == [1.0, 2.0, 3.0]


def test_fast_qkv_coordinate_mapping() -> None:
    config = SimpleNamespace(
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        hidden_size=4,
        attention_output_gate=False,
    )
    mapping = _mapping(
        "QKVMapping",
        hf_param={"q": "q", "k": "k", "v": "v"},
        tp_rank=1,
        _get_config=lambda _module: config,
    )
    # Global packed rows are [q0, q1, k0, v0, q2, q3, k1, v1], with
    # two scalar rows per head. TP rank 1 owns the second four heads.
    record = _fast_record(mapping, (8, 4))
    record.slots = [("q", (8, 4)), ("k", (4, 4)), ("v", (4, 4))]
    local_rows = torch.tensor([0, 2, 4, 6])
    local_indices = local_rows * 4 + 1

    _slots, counts, indices, values = export.sparse_hf_entry(
        record,
        local_indices,
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
        {},
    )

    assert counts.tolist() == [2, 1, 1]
    assert indices.tolist() == [17, 25, 9, 9]
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]


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
