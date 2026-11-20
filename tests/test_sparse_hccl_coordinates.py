"""CPU differential checks against the actual native VIME HF converter."""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _unit_stubs

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def native():
    names = (
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
        "ray",
        "ray.actor",
        "vime.utils.model_config",
    )
    saved = _unit_stubs.save_sys_modules(names)
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    config = types.ModuleType("vime.utils.model_config")
    config.load_hf_config = lambda *a, **k: pytest.fail("HF config loading is outside coordinate conversion")
    sys.modules[config.__name__] = config
    try:
        probe = importlib.import_module("vime.backends.megatron_utils.delta_sync.sparse_hf_export")
        common = importlib.import_module("vime.backends.megatron_utils.update_weight.common")
        converter = importlib.import_module("vime.backends.megatron_utils.megatron_to_hf")
        yield probe, common, converter
    finally:
        _unit_stubs.restore_sys_modules(saved)


CASES = [
    ("embedding.word_embeddings.weight", (8, 4), 0, None),
    ("output_layer.weight", (8, 4), 0, None),
    ("decoder.final_layernorm.weight", (4,), None, None),
    ("decoder.layers.3.self_attention.linear_qkv.weight", (16, 4), 0, None),
    ("decoder.layers.3.self_attention.linear_qkv.bias", (16,), 0, None),
    ("decoder.layers.3.self_attention.linear_proj.weight", (4, 8), 1, None),
    ("decoder.layers.3.mlp.linear_fc1.weight", (16, 4), 0, None),
    ("decoder.layers.3.mlp.experts.linear_fc1.weight7", (16, 4), 0, None),
    ("decoder.layers.3.mlp.experts.linear_fc2.weight7", (4, 8), 0, None),
    ("decoder.layers.3.self_attention.q_layernorm.weight", (2,), None, None),
    ("decoder.layers.3.mlp.linear_fc1.weight", (16, 4), 0, (8, 8)),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("name,shape,partition_dim,sections", CASES)
def test_sparse_coordinates_equal_full_native_conversion(native, dtype, tp_size, name, shape, partition_dim, sections):
    probe_module, common, converter = native
    args = SimpleNamespace(vocab_size=7, kv_channels=2, hidden_size=4, num_attention_heads=4, num_query_groups=2)
    name = "module.module." + name
    # Grouped expert FC2 records dimension zero but is actually column-sharded.
    dim = 1 if "linear_fc2.weight" in name and partition_dim == 0 else partition_dim
    if dim is not None:
        assert shape[dim] % tp_size == 0
    complete = (torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float() / 8).to(dtype)
    if dim is None:
        shards = [complete.clone()]
        effective_tp = 1
    elif "linear_fc1.weight" in name:
        gate, up = complete.chunk(2, dim=0)
        shards = [torch.cat(pair) for pair in zip(gate.chunk(tp_size), up.chunk(tp_size), strict=True)]
        effective_tp = tp_size
    else:
        shards = [part.clone() for part in complete.chunk(tp_size, dim=dim)]
        effective_tp = tp_size
    attrs = {
        "tensor_model_parallel": dim is not None,
        "partition_dim": partition_dim,
        "partition_stride": 1,
        "partition_sections": sections,
    }
    infos = [SimpleNamespace(name=name, attrs=attrs, shape=s.shape) for s in shards]
    expected_before = dict(converter.convert_to_hf(args, "qwen3moe", name, complete, None))
    actual = {key: value.clone() for key, value in expected_before.items()}
    changed_shards = []
    emitted = 0
    for rank, (shard, info) in enumerate(zip(shards, infos, strict=True)):
        # Include boundaries, zero values, negative zero and sparse interior changes.
        flat = shard.flatten().clone()
        positions = torch.unique(torch.tensor([0, flat.numel() // 2, flat.numel() - 1]))
        flat[positions] = -0.0
        if positions.numel() > 1:
            flat[positions[-1]] = -3.0
        changed_shards.append(flat.reshape(shard.shape))
        sparse = torch.full_like(shard, float("nan"))
        sparse.flatten()[positions] = flat[positions]
        before_probe = sparse.clone()
        probe = probe_module.SparseHFProbe(args, "qwen3moe", info, rank, effective_tp)
        for key, value in probe.megatron_to_hf(sparse).items():
            mask = ~torch.isnan(value)
            actual[key][mask] = value[mask]
            emitted += mask.sum().item()
        integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
        assert torch.equal(sparse.view(integer), before_probe.view(integer))
    rebuilt = (
        changed_shards[0]
        if dim is None
        else common._merge_param_partitions(name, changed_shards, partition_dim, sections)
    )
    expected = dict(converter.convert_to_hf(args, "qwen3moe", name, rebuilt, None))
    assert actual.keys() == expected.keys()
    assert emitted > 0
    integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
    for key in expected:
        assert actual[key].shape == expected[key].shape
        assert torch.equal(actual[key].contiguous().view(integer), expected[key].contiguous().view(integer)), key


def test_no_bridge_dependency_or_collective_in_probe(native, monkeypatch):
    probe_module, common, converter = native
    monkeypatch.setitem(sys.modules, "megatron.bridge", None)
    monkeypatch.setattr(torch.distributed, "all_gather", lambda *a, **k: pytest.fail("dense gather"))
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: pytest.fail("dense broadcast"))
    args = SimpleNamespace(vocab_size=4)
    info = SimpleNamespace(
        name="module.module.embedding.word_embeddings.weight",
        attrs={"tensor_model_parallel": True, "partition_dim": 0},
    )
    result = probe_module.SparseHFProbe(args, "qwen3moe", info, 1, 2).megatron_to_hf(torch.ones(2, 4))
    value = result["model.embed_tokens.weight"]
    assert torch.isnan(value[:2]).all() and torch.equal(value[2:], torch.ones(2, 4))
    empty = probe_module.SparseHFProbe(args, "qwen3moe", info, 1, 2).megatron_to_hf(torch.full((2, 4), float("nan")))
    assert torch.isnan(empty["model.embed_tokens.weight"]).all()


@pytest.mark.parametrize("invalid", ["duplicate", "dtype"])
def test_ambiguous_or_dtype_changing_converter_is_rejected(native, monkeypatch, invalid):
    probe_module, _, converter = native
    output = (
        [("hf", torch.ones(2)), ("hf", torch.zeros(2))] if invalid == "duplicate" else [("hf", torch.ones(2).half())]
    )
    monkeypatch.setattr(converter, "convert_to_hf", lambda *a, **k: output)
    info = SimpleNamespace(name="weight", attrs={})
    with pytest.raises(ValueError, match="Unsupported sparse conversion output"):
        probe_module.SparseHFProbe(SimpleNamespace(), "qwen3moe", info, 0, 1).megatron_to_hf(torch.ones(2))


@pytest.mark.parametrize(
    "expert,sharded,tp_rank,tp_size,replica,rank,owner,local,expected",
    [
        (False, True, 1, 2, 0, 1, 1, True, True),
        (False, False, 1, 2, 0, 1, 1, True, False),
        (False, True, 0, 2, 1, 0, 0, True, False),
        (False, True, 0, 2, 0, 2, 2, False, False),
        (True, True, 0, 1, 0, 1, 1, True, True),
        (True, True, 1, 2, 0, 3, 3, True, True),
        (True, False, 1, 2, 0, 3, 3, True, False),
        (False, True, 0, 2, 0, 2, 0, True, False),
    ],
)
def test_directory_keeps_tp_ep_shards_and_deduplicates_replicas(
    native,
    monkeypatch,
    expert,
    sharded,
    tp_rank,
    tp_size,
    replica,
    rank,
    owner,
    local,
    expected,
):
    export = importlib.import_module("vime.backends.megatron_utils.delta_sync.megatron_delta_export")
    mpu = sys.modules["megatron.core.parallel_state"]
    monkeypatch.setattr(sys.modules["megatron.core"], "parallel_state", mpu)
    for prefix in ("tensor_model", "expert_tensor"):
        monkeypatch.setattr(mpu, f"get_{prefix}_parallel_rank", lambda: tp_rank, raising=False)
        monkeypatch.setattr(mpu, f"get_{prefix}_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda **kw: replica, raising=False)
    monkeypatch.setattr(mpu, "get_expert_data_parallel_rank", lambda: replica, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        export, "_exchange_slot_tables", lambda rows: [setattr(r, "slots", [("hf", (4,))]) for r in rows]
    )
    name = "mlp.experts.linear_fc1.weight7" if expert else "embedding.word_embeddings.weight"
    info = SimpleNamespace(
        name=name, dtype=torch.float32, src_rank=owner, attrs={"tensor_model_parallel": sharded, "partition_dim": 0}
    )
    iterator = SimpleNamespace(
        args=SimpleNamespace(),
        model_name="qwen3moe",
        hf_vocab_size=4,
        param_infos=[info],
        local_weights=lambda: {name: torch.zeros(4)} if local else {},
    )
    (record,) = export.build_export_index(iterator, [])
    assert record.contributes is expected
    assert (record.param is not None) is local


@pytest.mark.parametrize(
    "options",
    [
        {"quantization_config": {"quant_method": "fp8"}},
        {"dspark_enabled": True},
        {"enable_mtp_training": True},
    ],
)
def test_unsupported_conversion_rejected_before_initializing_iterator(native, options):
    probe_module, _, _ = native
    options = options.copy()
    quantization = options.pop("quantization_config", None)
    with pytest.raises(ValueError):
        probe_module.SparseHFExporter(
            SimpleNamespace(**options), [], model_name="qwen3moe", quantization_config=quantization
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
