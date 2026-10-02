"""Same-job SimpleStorage lifecycle, with a separate lease coordinator actor."""

import asyncio
import hashlib
import json
import time
import uuid
from argparse import Namespace
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import ray

from vime.rollout.transfer_queue_adapter import EncodedGroup, SimpleStorageAdapter, encode_group
from vime.rollout.transfer_queue_coordinator import QueueCoordinator
from vime.utils.async_utils import run
from vime.utils.types import Sample


class TransferQueueRuntime:
    def __init__(self, args: Namespace):
        if version("TransferQueue") != "0.1.9":
            raise ValueError("VIME SimpleStorage adapter requires TransferQueue 0.1.9")
        from omegaconf import OmegaConf
        from transfer_queue.client import AsyncTransferQueueClient
        from transfer_queue.controller import TransferQueueController
        from transfer_queue.storage.simple_storage import SimpleStorageUnit

        self.args = args
        self.job_id = args.transfer_queue_job_id
        self.restart_epoch = args.transfer_queue_restart_epoch
        self.namespace = f"vime-tq-{self.job_id}-{self.restart_epoch}"
        recovery = None
        self.sampling_digest = hashlib.sha256(
            json.dumps(
                {
                    "temperature": args.rollout_temperature,
                    "top_p": args.rollout_top_p,
                    "top_k": args.rollout_top_k,
                    "max_response_len": args.rollout_max_response_len,
                    "group_size": args.n_samples_per_prompt,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if args.start_rollout_id:
            if not args.load:
                raise ValueError("queue recovery requires a common checkpoint")
            manifest = json.loads((Path(args.load) / "transfer-queue.json").read_text())
            if (
                manifest["next_rollout_id"] != args.start_rollout_id
                or manifest["sampling_digest"] != self.sampling_digest
            ):
                raise ValueError("queue checkpoint and data/training cursor disagree")
            if args.transfer_queue_restart_epoch <= manifest["restart_epoch"] or self.job_id != manifest["job_id"]:
                raise ValueError("recovery requires the same job and a new restart epoch")
            recovery = manifest["queue"]

        self.controller = TransferQueueController.options(name="controller", namespace=self.namespace).remote(
            polling_mode=True
        )
        self.storage = SimpleStorageUnit.options(name="storage", namespace=self.namespace).remote(
            storage_unit_size=args.transfer_queue_max_groups * args.n_samples_per_prompt
        )
        storage_info, info = ray.get(
            [self.storage.get_zmq_server_info.remote(), self.controller.get_zmq_server_info.remote()],
            timeout=args.transfer_queue_timeout_s,
        )
        config = OmegaConf.create({"zmq_info": {"storage": storage_info}}, flags={"allow_objects": True})
        self.client = AsyncTransferQueueClient(f"client-{self.namespace}", info)
        self.client.initialize_storage_manager("SimpleStorage", config)
        self.adapter = SimpleStorageAdapter(self.client, args.transfer_queue_timeout_s)
        self.coordinator = (
            ray.remote(QueueCoordinator)
            .options(num_cpus=0.1, name="coordinator", namespace=self.namespace)
            .remote(
                args.transfer_queue_max_groups,
                args.transfer_queue_max_tokens,
                args.transfer_queue_max_bytes,
                args.transfer_queue_lease_s,
            )
        )
        self.policy_version: str | None = None
        self.paused = True
        self.batch_id: str | None = None
        self.current: list[EncodedGroup] = []
        self.decoded: dict[str, list[Sample]] = {}
        self.rollout_id: int | None = None
        self.transfer_lock = asyncio.Lock()
        self.metrics = {
            "put_groups": 0,
            "get_groups": 0,
            "payload_bytes": 0,
            "put_s": 0.0,
            "get_s": 0.0,
            "gc_failures": 0,
            "peak_working_bytes": 0,
        }
        if recovery is not None:
            ray.get(self.coordinator.restore.remote(recovery), timeout=args.transfer_queue_timeout_s)

    def confirm_publication(self, engines) -> str:
        versions = ray.get(
            [engine.get_weight_version.remote() for engine in engines], timeout=self.args.transfer_queue_timeout_s
        )
        if not versions or any(not isinstance(v, str) or not v for v in versions) or len(set(versions)) != 1:
            self.paused = True
            raise RuntimeError("serving cohort has not acknowledged one policy version")
        if self.current:
            raise RuntimeError("queue payload cleanup has not completed")
        if self.batch_id is not None and versions[0] == self.policy_version:
            raise RuntimeError("optimizer update was not published as a new serving policy")
        ray.get(self.coordinator.publication_confirmed.remote(), timeout=self.args.transfer_queue_timeout_s)
        self.policy_version, self.paused = versions[0], False
        self.decoded.clear()
        return versions[0]

    def begin_round(self, rollout_id: int) -> None:
        if self.paused or self.policy_version is None or self.current:
            raise RuntimeError("queue producer is paused or its previous round is incomplete")
        self.rollout_id = rollout_id
        self.batch_id = f"{self.namespace}/round-{rollout_id}/{uuid.uuid4().hex}"

    def _encode(self, samples: list[Sample]) -> EncodedGroup:
        if self.rollout_id is None:
            raise ValueError("queue round has not started")
        return encode_group(
            samples,
            job_id=self.job_id,
            restart_epoch=self.restart_epoch,
            group_id=f"r{self.rollout_id}-g{samples[0].group_index}",
            attempt_id="0",
            expected_children=self.args.n_samples_per_prompt,
            policy_version=self.policy_version,
            sampling_digest=self.sampling_digest,
            reward_key=self.args.reward_key,
            collection_round=self.rollout_id,
        )

    async def accept_group(self, samples: list[Sample]) -> None:
        # One codec/write/read at a time bounds temporary buffers while other
        # generation tasks keep running. The control-plane actor remains separate.
        completed = False
        try:
            async with self.transfer_lock:
                await self._accept_group(samples)
            completed = True
        finally:
            if not completed:
                self.paused = True

    async def _accept_group(self, samples: list[Sample]) -> None:
        if self.paused or self.policy_version is None:
            raise RuntimeError("queue producer is paused")
        group = self._encode(samples)
        key = group.manifest.payload_ref
        reserved = await asyncio.wait_for(
            self.coordinator.reserve.remote(replace(group, rows=())), self.args.transfer_queue_timeout_s
        )
        if not reserved:
            if key not in self.decoded:
                raise RuntimeError("a previous transfer has not completed")
            return
        self.current.append(group)
        self.metrics["peak_working_bytes"] = max(
            self.metrics["peak_working_bytes"], sum(item.working_bytes for item in self.current)
        )
        complete = False
        try:
            started = time.monotonic()
            metadata = await self.adapter.put(group)
            self.metrics["put_s"] += time.monotonic() - started
            self.metrics["put_groups"] += 1
            self.metrics["payload_bytes"] += group.manifest.byte_count
            started = time.monotonic()
            decoded = await self.adapter.get(metadata, group)
            self.metrics["get_s"] += time.monotonic() - started
            self.metrics["get_groups"] += 1
            await asyncio.wait_for(self.coordinator.mark_readable.remote(key), self.args.transfer_queue_timeout_s)
            self.decoded[key] = decoded
            complete = True
        finally:
            if not complete:
                self.paused = True
                # A failed put may have written rows. Reclaim this owned partition
                # before releasing capacity; keep the original failure if GC fails.
                try:
                    await self.adapter.delete(group)
                    await asyncio.wait_for(self.coordinator.rollback.remote(key), self.args.transfer_queue_timeout_s)
                    self.current.remove(group)
                except (TimeoutError, RuntimeError, ray.exceptions.RayError):
                    self.metrics["gc_failures"] += 1

    def roundtrip(self, samples: list[Sample], rollout_id: int) -> list[Sample]:
        if self.rollout_id != rollout_id:
            raise ValueError("queue collection and training round disagree")
        grouped: dict[int, list[Sample]] = {}
        for sample in samples:
            if sample.group_index is None:
                raise ValueError("queue groups require the original prompt/draw identity")
            grouped.setdefault(sample.group_index, []).append(sample)
        admitted = {group.manifest.payload_ref: group for group in self.current}
        keys = []
        for samples in grouped.values():
            group = self._encode(samples)
            key = group.manifest.payload_ref
            if key not in admitted or admitted[key] != group:
                raise ValueError("a trajectory changed after queue publication")
            keys.append(key)
        if len(keys) != self.args.rollout_batch_size or set(self.decoded) != set(keys):
            raise ValueError("training batch differs from the completed queue groups")
        self.paused = True
        ray.get(
            self.coordinator.seal.remote(tuple(keys), self.batch_id, self.policy_version),
            timeout=self.args.transfer_queue_timeout_s,
        )
        return [sample for key in keys for sample in self.decoded[key]]

    def training_complete(self, rollout_id: int) -> None:
        ray.get(
            self.coordinator.finish_training.remote(rollout_id, self.batch_id),
            timeout=self.args.transfer_queue_timeout_s,
        )

        self.reclaim_committed()

    def reclaim_committed(self) -> None:
        """Retry owned payload GC without acknowledging or replaying training."""

        async def cleanup():
            for group in tuple(self.current):
                await self.adapter.delete(group)
                await asyncio.wait_for(
                    self.coordinator.reclaimed.remote(group.manifest.payload_ref), self.args.transfer_queue_timeout_s
                )
                self.current.remove(group)

        try:
            run(cleanup())
        except (TimeoutError, RuntimeError, ray.exceptions.RayError):
            self.metrics["gc_failures"] += 1
            raise

    def save(self, rollout_id: int) -> None:
        snapshot = ray.get(self.coordinator.snapshot.remote(), timeout=self.args.transfer_queue_timeout_s)
        if snapshot["committed_round"] != rollout_id:
            raise ValueError("checkpoint and committed training round disagree")
        payload = {
            "queue": snapshot,
            "job_id": self.job_id,
            "restart_epoch": self.restart_epoch,
            "next_rollout_id": rollout_id + 1,
            "sampling_digest": self.sampling_digest,
            "policy_version": self.policy_version,
        }
        path = Path(self.args.save) / "transfer-queue.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True))
        temporary.replace(path)

    def close(self) -> None:
        self.paused = True
        self.client.close()
