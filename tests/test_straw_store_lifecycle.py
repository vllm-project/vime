"""Reader lifetimes and scratch cleanup against the real Straw store."""

from types import SimpleNamespace

import pytest
import torch
from straw.store import SharedFilesystemStore

from vime.data.codec import CODECS, SampleCodec
from vime.data.tensor import TensorRef
from vime.data.transport import DiskPayloadRef, close_rollout_stores, rollout_store
from vime.utils.external_utils import command_utils as U
from vime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.parametrize("decode_failure", [False, True])
def test_payload_read_closes_its_store_without_invalidating_lazy_references(monkeypatch, tmp_path, decode_failure):
    pool = tmp_path / "pool"
    with SharedFilesystemStore(pool, "reader-test", codecs=CODECS) as writer:
        expected = torch.arange(8, dtype=torch.int32).reshape(2, 4)
        ref = SampleCodec(writer).publish(
            Sample(index=1, tokens=[1, 2, 3], rollout_routed_experts=expected), submission_id="sample"
        )
        closed = []

        class TrackedStore(SharedFilesystemStore):
            def close(self):
                super().close()
                closed.append(self)

        monkeypatch.setattr("straw.store.SharedFilesystemStore", TrackedStore)
        payload = DiskPayloadRef(ref, str(pool))
        if decode_failure:

            def fail(*args, **kwargs):
                raise ValueError("decode failed")

            monkeypatch.setattr(SampleCodec, "load", fail)
            with pytest.raises(ValueError, match="decode failed"):
                payload.load()
        else:
            sample = payload.load()
            assert len(closed) == 1
            assert isinstance(sample.rollout_routed_experts, TensorRef)
            # Use a different live reader after the payload reader has closed.
            torch.testing.assert_close(sample.rollout_routed_experts.load(reader=writer), expected)
        assert len(closed) == 1


def test_scratch_closes_only_its_cached_writers_and_allows_new_incarnations(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "get_test_tmp_dir", lambda: str(tmp_path))

    def args(root):
        return SimpleNamespace(rollout_data_transport="straw", rollout_data_dir=str(root))

    with U.test_work_directory("owner_", keep=False) as scratch:
        owned = args(scratch / "pool")
        sibling = args(tmp_path / (scratch.name + "-other"))
        first, codec, _ = rollout_store(owned)
        saved = codec.publish({"value": 1}, submission_id="before-close")
        other, other_codec, _ = rollout_store(sibling)
        try:
            close_rollout_stores(root=scratch)
            second, codec, _ = rollout_store(owned)
            assert second is not first
            assert codec.load(saved) == {"value": 1}
            assert rollout_store(sibling)[0] is other
            other_saved = other_codec.publish({"value": 2}, submission_id="still-open")
            assert other_codec.load(other_saved) == {"value": 2}
        finally:
            close_rollout_stores(root=sibling.rollout_data_dir)
    assert not scratch.exists()
    assert (tmp_path / (scratch.name + "-other")).exists()
