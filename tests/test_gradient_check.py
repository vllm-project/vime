"""CPU regression tests for full-tensor parallel replay diagnostics."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "vime/backends/megatron_utils/gradient_check.py"


@pytest.fixture(scope="module")
def checker():
    assert MODULE_PATH.exists(), "parameter-wise gradient checker is missing"
    spec = importlib.util.spec_from_file_location("gradient_check", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(value, *, shape=None, start=0, tp_rank=0, tp_size=1, dim=0, stride=1, glu=False):
    return dict(
        grad=value.flatten(),
        shape=tuple(shape or value.shape),
        start=start,
        tp_rank=tp_rank,
        tp_size=tp_size,
        partition_dim=dim,
        partition_stride=stride,
        glu=glu,
    )


def write_snapshot(path, ranks):
    path.mkdir(parents=True, exist_ok=True)
    for rank, tensors in enumerate(ranks):
        torch.save(dict(version=1, rank=rank, world_size=len(ranks), tensors=tensors), path / f"rank-{rank}.pt")
    return path


def test_equal_norm_direction_error_is_detected(checker, tmp_path):
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = -expected
    assert math.isclose(actual.norm().item(), expected.norm().item(), rel_tol=0.01, abs_tol=0.01)
    left = write_snapshot(tmp_path / "left", [{"decoder.layers.0.weight": record(expected)}])
    right = write_snapshot(tmp_path / "right", [{"decoder.layers.0.weight": record(actual)}])
    with pytest.raises(AssertionError, match=r"decoder.layers.0.weight.*index=.*rank"):
        checker.compare_snapshots(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dim", [0, 1])
def test_tp_and_dp_resharding(checker, tmp_path, dim):
    value = torch.arange(24.0).reshape(6, 4)
    shards = value.chunk(2, dim=dim)
    ranks = []
    for tp, shard in enumerate(shards):
        flat = shard.flatten()
        for start, stop in [(0, 5), (5, flat.numel())]:
            ranks.append(
                {"weight": record(flat[start:stop], shape=shard.shape, start=start, tp_rank=tp, tp_size=2, dim=dim)}
            )
    left = write_snapshot(tmp_path / "left", [{"weight": record(value)}])
    right = write_snapshot(tmp_path / "right", ranks)
    assert checker.compare_snapshots(left, right, rtol=0, atol=0) == 1


def test_glu_reorders_gate_and_up_projections(checker, tmp_path):
    value = torch.arange(32.0).reshape(8, 4)
    gate, up = value.chunk(2)
    ranks = [
        {
            "decoder.layers.3.mlp.linear_fc1.weight": record(
                torch.cat((gate.chunk(2)[tp], up.chunk(2)[tp])), tp_rank=tp, tp_size=2, stride=2, glu=True
            )
        }
        for tp in range(2)
    ]
    left = write_snapshot(tmp_path / "left", [{"decoder.layers.3.mlp.linear_fc1.weight": record(value)}])
    right = write_snapshot(tmp_path / "right", ranks)
    assert checker.compare_snapshots(left, right, rtol=0, atol=0) == 1


def test_pipeline_stages_and_replicas(checker, tmp_path):
    value = torch.ones(4)
    left = write_snapshot(tmp_path / "left", [{f"decoder.layers.{i}.norm": record(value) for i in range(2)}])
    right = write_snapshot(
        tmp_path / "right",
        [
            {"decoder.layers.0.norm": record(value)},
            {"decoder.layers.0.norm": record(value)},
            {"decoder.layers.1.norm": record(value)},
            {"decoder.layers.1.norm": record(value)},
        ],
    )
    assert checker.compare_snapshots(left, right, rtol=0, atol=0) == 2


@pytest.mark.parametrize("fault", ["gap", "replica", "nonfinite", "missing_rank", "missing_parameter", "shape"])
def test_incomplete_or_invalid_snapshots_fail(checker, tmp_path, fault):
    left = write_snapshot(tmp_path / "left", [{"weight": record(torch.ones(4))}])
    ranks = [{"weight": record(torch.ones(4))}]
    if fault == "gap":
        ranks[0]["weight"] = record(torch.ones(2), shape=(4,))
    elif fault == "replica":
        ranks.append({"weight": record(torch.zeros(4))})
    elif fault == "nonfinite":
        ranks[0]["weight"]["grad"][0] = float("nan")
    elif fault == "missing_parameter":
        ranks[0] = {"other": record(torch.ones(4))}
    elif fault == "shape":
        ranks[0]["weight"] = record(torch.ones(2, 2))
    right = write_snapshot(tmp_path / "right", ranks)
    if fault == "missing_rank":
        payload = torch.load(right / "rank-0.pt", weights_only=True)
        payload["world_size"] = 2
        torch.save(payload, right / "rank-0.pt")
    with pytest.raises((AssertionError, ValueError)):
        checker.compare_snapshots(left, right, rtol=1e-5, atol=1e-6)


def test_tolerance_and_zero_values(checker, tmp_path):
    value = torch.tensor([0.0, 1.0])
    left = write_snapshot(tmp_path / "left", [{"weight": record(value)}])
    right = write_snapshot(tmp_path / "right", [{"weight": record(value + 1e-7)}])
    assert checker.compare_snapshots(left, right, rtol=1e-5, atol=1e-6) == 1
    with pytest.raises(AssertionError):
        checker.compare_snapshots(left, right, rtol=0, atol=0)


def test_empty_snapshot_does_not_pass(checker, tmp_path):
    left = write_snapshot(tmp_path / "left", [{}])
    with pytest.raises(ValueError, match="empty"):
        checker.compare_snapshots(left, left, rtol=0, atol=0)


@pytest.mark.parametrize("stride", [1, 2])
def test_glu_stride_metadata_does_not_change_layout(checker, tmp_path, stride):
    value = torch.arange(8.0).reshape(8, 1)
    ranks = []
    for tp in range(2):
        local = torch.cat([value[:4].chunk(2)[tp], value[4:].chunk(2)[tp]])
        ranks.append({"linear_fc1.weight": record(local, tp_rank=tp, tp_size=2, stride=stride, glu=True)})
    left = write_snapshot(tmp_path / "left", [{"linear_fc1.weight": record(value, glu=True)}])
    right = write_snapshot(tmp_path / "right", ranks)
    assert checker.compare_snapshots(left, right, rtol=0, atol=0) == 1


def test_nongated_fc1_is_concatenated_normally(checker, tmp_path):
    value = torch.arange(8.0).reshape(4, 2)
    ranks = [{"linear_fc1.weight": record(shard, tp_rank=tp, tp_size=2)} for tp, shard in enumerate(value.chunk(2))]
    left = write_snapshot(tmp_path / "left", [{"linear_fc1.weight": record(value)}])
    right = write_snapshot(tmp_path / "right", ranks)
    assert checker.compare_snapshots(left, right, rtol=0, atol=0) == 1


def test_empty_dp_intersection_is_allowed(checker, tmp_path):
    value = torch.ones(2)
    left = write_snapshot(tmp_path / "left", [{"weight": record(value)}])
    right = write_snapshot(
        tmp_path / "right", [{"weight": record(value)}, {"weight": record(torch.empty(0), shape=(2,), start=2)}]
    )
    assert checker.compare_snapshots(left, right, rtol=0, atol=0) == 1


def test_missing_step_is_rejected(checker, tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    write_snapshot(left / "actor/rollout-0/step-0", [{"weight": record(torch.ones(2))}])
    write_snapshot(right / "actor/rollout-0/step-1", [{"weight": record(torch.ones(2))}])
    with pytest.raises(ValueError, match="steps differ"):
        checker.compare_runs(left, right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_tolerance_is_rejected(checker, tmp_path, value):
    with pytest.raises(ValueError, match="Tolerances"):
        checker.compare_snapshots(tmp_path, tmp_path, rtol=value)


@pytest.mark.parametrize("bucket_offset", [0, 128])
def test_capture_reads_only_reduced_dp_slice_and_preserves_buffers(checker, tmp_path, monkeypatch, bucket_offset):
    import sys
    from types import ModuleType, SimpleNamespace

    param = torch.nn.Parameter(torch.zeros(2, 3))
    param.tensor_model_parallel = False
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    common = ModuleType("vime.backends.megatron_utils.update_weight.common")
    common.named_params_and_buffers = lambda args, model: [("weight", param), ("frozen", frozen)]
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    config = SimpleNamespace(num_distributed_optimizer_instances=1, use_distributed_optimizer=True)
    for rank in range(2):
        # Poison the unreduced region: reading the whole main_grad would silently
        # include it. The valid DP ranges straddle the parameter's boundary.
        data = torch.full((8,), -999.0)
        if rank == 0:
            data[1:4] = torch.tensor([10.0, 11.0, 12.0])
        else:
            data[4:7] = torch.tensor([13.0, 14.0, 15.0])
        before = data.clone()
        bucket = SimpleNamespace(grad_data=data, offset=bucket_offset, params_list=[param])
        buffer = SimpleNamespace(
            buckets=[bucket],
            data_parallel_group=None,
            param_index_map={param: (1 + bucket_offset, 7 + bucket_offset, 0)},
        )
        chunk = SimpleNamespace(ddp_config=config, buffers=[buffer])
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None, r=rank: r)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
        checker.save_gradient_snapshot(SimpleNamespace(), [chunk], tmp_path / "captured")
        assert torch.equal(before, data)
    expected = write_snapshot(tmp_path / "expected", [{"weight": record(torch.arange(10.0, 16.0).reshape(2, 3))}])
    assert checker.compare_snapshots(expected, tmp_path / "captured", rtol=0, atol=0) == 1


def test_compare_runs_checks_all_steps(checker, tmp_path):
    for step in ("actor/rollout-0/step-0", "actor/rollout-0/step-1", "critic/rollout-1/step-0"):
        for side in ("left", "right"):
            write_snapshot(tmp_path / side / step, [{"weight": record(torch.ones(2))}])
    assert checker.compare_runs(tmp_path / "left", tmp_path / "right") == 3


def test_training_hook_runs_after_backward_before_optimizer_preparation(tmp_path, monkeypatch):
    """Execute the real step body with training work replaced by event recorders."""
    import ast
    import sys
    from types import ModuleType, SimpleNamespace

    source = (MODULE_PATH.parent / "model.py").read_text()
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "train_one_step"
    )
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ast.fix_missing_locations(tree)
    events = []
    snapshot_module = ModuleType("vime.backends.megatron_utils.gradient_check")
    paths = []

    def snapshot(args, model, directory):
        events.append("snapshot")
        paths.append(directory)

    snapshot_module.save_gradient_snapshot = snapshot
    monkeypatch.setitem(sys.modules, snapshot_module.__name__, snapshot_module)
    args = SimpleNamespace(
        ci_save_parameter_grads=str(tmp_path),
        custom_megatron_before_train_step_hook_path=None,
        seq_length=16,
        micro_batch_size=1,
        decoder_seq_length=None,
        check_for_nan_in_loss_and_grad=False,
        ci_test=False,
    )
    chunk = SimpleNamespace(role="critic", zero_grad_buffer=lambda: events.append("zero"))

    def backward(**kwargs):
        events.append("backward")
        return []

    def prepare():
        events.append("prepare")
        return False

    def step():
        events.append("step")
        return True, 1.0, 0

    optimizer = SimpleNamespace(
        zero_grad=lambda: events.append("optimizer_zero"), prepare_grads=prepare, get_grad_norm=lambda: 1.0, step=step
    )
    scheduler = SimpleNamespace(step=lambda **kwargs: events.append("schedule"))
    namespace = dict(
        get_args=lambda: args,
        get_forward_backward_func=lambda: backward,
        torch=torch,
        math=math,
        _wrap_forward_step_with_microbatch_pbar=lambda forward, bar: forward,
        mpu=SimpleNamespace(is_pipeline_last_stage=lambda **kwargs: False),
    )
    exec(compile(tree, "model.py", "exec"), namespace)
    namespace["train_one_step"](args, 3, 2, [], [chunk], optimizer, scheduler, 1, 4)
    assert events == [
        "zero",
        "optimizer_zero",
        "backward",
        "snapshot",
        "prepare",
        "step",
        "schedule",
        "zero",
        "optimizer_zero",
    ]
    assert paths == [tmp_path / "critic/rollout-3/step-2"]
    events.clear()
    args.ci_save_parameter_grads = None
    namespace["train_one_step"](args, 3, 2, [], [chunk], optimizer, scheduler, 1, 4)
    assert "snapshot" not in events
