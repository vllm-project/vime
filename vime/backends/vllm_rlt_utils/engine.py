"""Serial engine ownership and fail-closed full physical-weight publications."""

import hashlib
from pathlib import Path
from uuid import uuid4

import torch

from vime.utils.types import RecurrentTrace, Sample


def weight_digest(path: Path) -> tuple[str, list[Path]]:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights in {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), files


class NativeEngine:
    def init(self, args):
        from vllm_rlt import LLM, CacheConfig, ExecutionConfig, SchedulerConfig

        self.args = args
        self.epoch = uuid4().hex
        self.paused = True
        self.ready = False
        self.committed_digest: str | None = None
        self.llm = LLM(
            args.hf_checkpoint,
            device="cuda",
            dtype=args.params_dtype,
            attention_backend=args.rlt_attention_backend,
            execution_config=ExecutionConfig(cuda_graphs=args.rlt_cuda_graphs),
            cache_config=CacheConfig(num_blocks=args.rlt_kv_blocks),
            scheduler_config=SchedulerConfig(max_num_seqs=args.rlt_max_num_seqs),
        )

    def generate(self, samples: list[Sample], rollout_id: int) -> list[Sample]:
        from vllm_rlt import SamplingParams

        if self.paused or not self.ready:
            raise RuntimeError("Publish a complete policy and resume generation first")
        assert self.committed_digest is not None
        version = self.llm.get_weight_version()
        if any(sample.index is None for sample in samples):
            raise ValueError("Native rollout requires stable sample indices")
        params = [
            SamplingParams(
                max_tokens=self.args.rollout_max_response_len,
                temperature=self.args.rollout_temperature,
                min_loops=self.args.rlt_depth,
                max_loops=self.args.rlt_depth,
                seed=(self.args.seed + rollout_id * 1_000_003 + sample.index) % (2**63),
                logprobs=0,
                logprobs_mode="processed",
            )
            for sample in samples
        ]
        outputs = self.llm.generate([sample.tokens for sample in samples], params)
        for sample, output, sampling in zip(samples, outputs, params, strict=True):
            assert sampling.seed is not None
            response_length = len(output.token_ids)
            if (
                not output.finished
                or output.weight_version != version
                or output.log_probs is None
                or len(output.log_probs) != response_length
                or len(output.exit_depths) != response_length
                or not response_length
            ):
                raise RuntimeError("Incomplete or inconsistent native rollout output")
            sample.tokens += output.token_ids
            sample.response_length = response_length
            sample.rollout_log_probs = output.log_probs
            sample.weight_versions = [str(version)]
            sample.loss_mask = [1] * response_length
            sample.status = Sample.Status.TRUNCATED if output.finish_reason == "length" else Sample.Status.COMPLETED
            sample.recurrent_trace = RecurrentTrace(
                schema_version=1,
                model_family=self.args.rlt_model_family,
                model_revision=self.args.rlt_model_revision,
                engine_revision=self.args.rlt_engine_revision,
                runtime_epoch=self.epoch,
                policy_version=version,
                publication_digest=self.committed_digest,
                request_id=output.request_id,
                seed=sampling.seed,
                prefill_depth=self.args.rlt_depth,
                decode_depths=output.exit_depths,
                temperature=sampling.temperature,
                finish_reason=output.finish_reason,
                latent_seed=sampling.latent_seed,
                latent_profile="like-init-cpu-f32-v1" if sampling.latent_seed is not None else None,
            )
        return samples

    def pause_generation(self):
        self.paused = True
        self.llm.pause_generation()

    def flush_cache(self):
        self.llm.reset_prefix_cache()

    def update_weights_from_disk(self, model_path: str, weight_version: str):
        from safetensors import safe_open

        if not self.paused:
            raise RuntimeError("Pause generation before publication")
        version = int(weight_version)
        digest, files = weight_digest(Path(model_path))
        if version == self.llm.get_weight_version():
            if not self.ready or digest != self.committed_digest:
                raise ValueError("Conflicting publication at the committed version")
            return version
        self.ready = False
        self.llm.start_weight_update(version)

        def physical_weights(weights):
            for name in weights.keys():
                value = weights.get_tensor(name)
                yield name, value

        for file in files:
            with safe_open(file, framework="pt", device="cpu") as weights:
                self.llm.update_weights(physical_weights(weights))
        self.llm.finish_weight_update()
        self.committed_digest = digest
        self.ready = True
        return self.llm.get_weight_version()

    def get_weight_version(self):
        return self.llm.get_weight_version()

    def continue_generation(self):
        if not self.ready:
            raise RuntimeError("No complete physical-weight publication")
        self.llm.resume_generation()
        self.paused = False

    def close(self):
        self.paused = True
        self.llm.close()
        del self.llm
        torch.cuda.empty_cache()

    def shutdown(self):
        import ray

        ray.actor.exit_actor()
