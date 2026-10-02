"""Real SimpleStorage transport and coordinator lifecycle; no GPU model here.

Run in an existing environment containing TransferQueue==0.1.9. Publication
and optimizer completion are explicit fixture events, not model E2E evidence.
"""

import asyncio
import json
import os
import uuid
from argparse import Namespace

import pytest
import ray

from vime.rollout.transfer_queue_runtime import TransferQueueRuntime
from vime.utils.async_utils import run
from vime.utils.types import Sample

NUM_GPUS = 0
pytestmark = pytest.mark.integration


def samples(group_id, policy="v3"):
    return [
        Sample(
            index=group_id * 2 + child,
            group_index=group_id,
            rollout_id=group_id * 2 + child,
            tokens=[10, 11, 12] + [13] * child,
            response="answer",
            response_length=1 + child,
            reward=float(child),
            loss_mask=[1] + [0] * child,
            weight_versions=[policy],
            rollout_log_probs=[-0.25] * (1 + child),
            status=Sample.Status.COMPLETED,
        )
        for child in range(2)
    ]


def test_native_streaming_groups_soak_and_failed_read_cleanup(tmp_path, monkeypatch):
    job = uuid.uuid4().hex[:12]
    ray.init(
        address="local",
        namespace=f"tq-test-{job}",
        num_cpus=8,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=256 * 1024 * 1024,
        _temp_dir=f"/tmp/vime-tq-{os.getpid()}",
    )
    args = Namespace(
        transfer_queue_job_id=job,
        transfer_queue_restart_epoch=0,
        transfer_queue_max_groups=2,
        transfer_queue_max_tokens=100,
        transfer_queue_max_bytes=1 << 20,
        transfer_queue_timeout_s=30,
        transfer_queue_lease_s=60,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=16,
        reward_key=None,
        start_rollout_id=0,
        load=None,
        save=str(tmp_path),
    )
    queue = TransferQueueRuntime(args)
    recovered = None
    # Fixture publication event. The CUDA recipe must prove serving publication.
    queue.policy_version, queue.paused = "v3", False
    try:
        for round_id in range(101):
            queue.begin_round(round_id)
            first, second = samples(round_id * 2), samples(round_id * 2 + 1)

            async def write(first=first, second=second):
                await asyncio.gather(queue.accept_group(second), queue.accept_group(first))

            run(write())
            decoded = queue.roundtrip(first + second, round_id)
            assert [s.to_dict() for s in decoded] == [s.to_dict() for s in first + second]
            with pytest.raises(ray.exceptions.RayTaskError, match="quiescent"):
                ray.get(queue.coordinator.snapshot.remote())
            ray.get(queue.coordinator.start_training.remote())
            ray.get(queue.coordinator.heartbeat.remote())
            queue.training_complete(round_id)
            ray.get(queue.coordinator.publication_confirmed.remote())
            queue.paused = False
            queue.decoded.clear()
            assert not run(queue.client.async_get_partition_list())

        queue.save(100)
        assert ray.get(queue.coordinator.snapshot.remote())["committed_round"] == 100
        recovered = TransferQueueRuntime(
            Namespace(
                **(vars(args) | {"transfer_queue_restart_epoch": 1, "start_rollout_id": 101, "load": str(tmp_path)})
            )
        )
        assert ray.get(recovered.coordinator.snapshot.remote())["committed_round"] == 100
        assert recovered.namespace != queue.namespace and recovered.paused
        queue.begin_round(101)
        original_get, original_put, original_delete = queue.adapter.get, queue.adapter.put, queue.adapter.delete

        async def failed_read(*_args):
            raise TimeoutError("injected read timeout")

        monkeypatch.setattr(queue.adapter, "get", failed_read)
        with pytest.raises(TimeoutError, match="injected read"):
            run(queue.accept_group(samples(300)))
        assert queue.paused and not queue.current
        assert queue.metrics["gc_failures"] == 0
        assert not run(queue.client.async_get_partition_list())
        assert ray.get(queue.coordinator.snapshot.remote())["committed_round"] == 100

        async def completed_put_then_failure(encoded):
            await original_put(encoded)
            raise RuntimeError("injected completed-write failure")

        monkeypatch.setattr(queue.adapter, "get", original_get)
        monkeypatch.setattr(queue.adapter, "put", completed_put_then_failure)
        queue.paused = False
        with pytest.raises(RuntimeError, match="completed-write"):
            run(queue.accept_group(samples(301)))
        assert queue.paused and not queue.current
        assert not run(queue.client.async_get_partition_list())

        monkeypatch.setattr(queue.adapter, "put", original_put)
        queue.paused = False
        first, second = samples(302), samples(303)
        run(queue.accept_group(first))
        run(queue.accept_group(second))
        queue.roundtrip(first + second, 101)
        ray.get(queue.coordinator.start_training.remote())

        async def failed_delete(_encoded):
            raise TimeoutError("injected committed-delete failure")

        monkeypatch.setattr(queue.adapter, "delete", failed_delete)
        with pytest.raises(TimeoutError, match="committed-delete"):
            queue.training_complete(101)
        assert queue.paused and len(queue.current) == 2 and queue.metrics["gc_failures"] == 1
        # Simulate the driver's fail-stop decision. Committed groups stay committed;
        # GC retry cannot trigger another training plan or resume publication.
        ray.get(queue.coordinator.fail.remote())
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(queue.coordinator.start_training.remote())
        monkeypatch.setattr(queue.adapter, "delete", original_delete)
        queue.reclaim_committed()
        assert not queue.current and not run(queue.client.async_get_partition_list())
        assert queue.metrics["peak_working_bytes"] <= args.transfer_queue_max_bytes
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(queue.coordinator.publication_confirmed.remote())
        print("TransferQueue transport metrics:", json.dumps(queue.metrics, sort_keys=True))
    finally:
        # Close only this client's sockets; no global TQ close or Ray process kill.
        queue.close()
        if recovered is not None:
            recovered.close()
