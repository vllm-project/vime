"""Verify sparse-update scheduling and fail-closed generation lifecycle."""

from types import SimpleNamespace

import pytest

from vime.backends.megatron_utils.update_weight import update_weight_from_sparse_hccl as module


@pytest.fixture
def updater(monkeypatch):
    instance = module.UpdateWeightFromSparseHCCL.__new__(module.UpdateWeightFromSparseHCCL)
    instance.args = SimpleNamespace(update_weight_delta_verify_every=2)
    instance._client = object()
    instance._failed = False
    instance._seeded = True
    instance._steady_updates = 0
    instance.weight_version = 1
    instance._begin_update = lambda: None
    instance._ensure_export_index = lambda: None
    instance._sync_update_state = lambda error: (error is not None, instance._payload_started)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    return instance


@pytest.mark.unit
def test_verify_schedule_checks_sparse_writes_without_dense_replay(updater):
    verification = []
    finishes = []
    updater._send_sparse_delta = lambda *, verify: (verification.append(verify) or (1, 24, 4))
    updater._send_dense = lambda **kwargs: pytest.fail("steady verification must not replay dense weights")
    updater._finish_update = lambda: finishes.append(updater.weight_version)
    for _ in range(3):
        updater.update_weights()
    assert verification == [False, True, False]
    assert finishes == [2, 3, 4]
    assert updater._steady_updates == 3


@pytest.mark.unit
def test_steady_update_adds_no_device_synchronization(updater, monkeypatch):
    monkeypatch.setattr(module.torch.accelerator, "synchronize", lambda: pytest.fail("unexpected synchronization"))
    updater._send_sparse_delta = lambda **kwargs: (1, 24, 4)
    updater._finish_update = lambda: None
    updater.update_weights()
    assert not any(key.endswith("_seconds") for key in updater.pop_metrics())


@pytest.mark.unit
def test_sparse_delta_reuses_dtype_bucket_without_changing_payload(updater, monkeypatch):
    updater.args.update_weight_buffer_size = 1024
    updater._index = object()
    updater._snapshots = {}
    created = []
    original_bucket = module._SparseFlushBucket

    def make_bucket(*args, **kwargs):
        bucket = original_bucket(*args, **kwargs)
        created.append(bucket)
        return bucket

    monkeypatch.setattr(module, "_SparseFlushBucket", make_bucket)
    monkeypatch.setattr(module, "checksum", lambda positions, values: 123)
    tensor = module.torch.tensor
    entries = [
        ([(name, [4])], "float32", tensor([1]),
         tensor([index], dtype=module.torch.int32), tensor([value]), None)
        for name, index, value in [("a", 1, 10.), ("b", 3, 20.)]
    ]
    monkeypatch.setattr(module, "iter_delta_entries", lambda *args, **kwargs: iter(entries))
    payloads = []
    updater._publish_flush = lambda flush, **kwargs: payloads.append(flush)
    assert updater._send_sparse_delta() == (1, 16, 2)
    assert len(created) == 1
    assert [param.name for param in payloads[0].params] == ["a", "b"]
    assert payloads[0].positions.view(module.torch.int32).tolist() == [1, 3]
    assert payloads[0].values.tolist() == [10., 20.]


@pytest.mark.unit
def test_gather_queue_reuses_workspace_between_flushes(monkeypatch):
    created, observed = [], []

    def make_workspace():
        workspace = object()
        created.append(workspace)
        return workspace

    def gather(indices, values, counts, **kwargs):
        observed.append(kwargs["workspace"])
        return [(indices, values)]

    monkeypatch.setattr(module, "GatherWorkspace", make_workspace)
    monkeypatch.setattr(module, "gather_slot_entries_to_rank0", gather)
    consumed = []
    queue = module._GatherQueue(1, 1024, True, lambda *args: consumed.append(args))
    group = object()
    for name in ["a", "b"]:
        queue.put(group, [(name, [4])], "float32", module.torch.tensor([1]),
                  module.torch.tensor([1], dtype=module.torch.int32), module.torch.tensor([10.]))
    assert len(created) == 1
    assert observed == [created[0], created[0]]
    assert [entry[0] for entry in consumed] == ["a", "b"]


@pytest.mark.unit
def test_partial_payload_failure_keeps_generation_paused_and_rejects_reuse(updater):
    def send(*, verify):
        updater._payload_started = True
        raise RuntimeError("receiver failed after partial apply")

    updater._send_sparse_delta = send
    updater._finish_update = lambda: pytest.fail("partial weights must not resume generation")
    updater._recover_before_payload = lambda: pytest.fail("partial apply has no rollback")
    with pytest.raises(RuntimeError, match="partial apply"):
        updater.update_weights()
    assert updater._failed
    with pytest.raises(RuntimeError, match="cannot be reused"):
        updater.update_weights()


@pytest.mark.unit
def test_pre_payload_failure_uses_recovery_instead_of_finish(updater):
    recovered = []

    def prepare():
        raise RuntimeError("export preparation failed")

    updater._ensure_export_index = prepare
    updater._recover_before_payload = lambda: recovered.append(True)
    updater._finish_update = lambda: pytest.fail("failed preparation must not finish an update")
    with pytest.raises(RuntimeError, match="preparation failed"):
        updater.update_weights()
    assert recovered == [True]
    assert not updater._failed


@pytest.mark.unit
def test_finish_failure_rejects_following_update(updater):
    updater._send_sparse_delta = lambda *, verify: (1, 24, 4)

    def finish():
        raise RuntimeError("finish RPC failed")

    updater._finish_update = finish
    with pytest.raises(RuntimeError, match="finish RPC failed"):
        updater.update_weights()
    assert updater._failed
