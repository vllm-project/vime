"""Bounded collect-only export of the target LM head input."""

import hashlib
import json
import os
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from vime.utils.draft_feature_contract import (
    DraftFeatureManifest,
    DraftSequence,
    build_token_map,
    normalize_weight_versions,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DraftFeatureCollector:
    def __init__(self, args: Namespace, model: torch.nn.Module, round_id: int, source_version: str):
        self.args = args
        self.round_id = round_id
        self.source_version = source_version
        target = model
        self.output_layer = target.output_layer
        self.head_weight = (
            target.shared_embedding_or_output_weight()
            if target.share_embeddings_and_output_weights
            else self.output_layer.weight
        )
        if self.output_layer._parameters.get("bias") is not None:
            raise ValueError("collect-only draft features require a bias-free LM head")
        self.root = Path(args.draft_feature_output_dir) / args.draft_feature_run_id / f"round-{round_id}"
        self.root.mkdir(parents=True, exist_ok=False)
        checkpoint = Path(args.hf_checkpoint)
        self.model_digest = _digest(checkpoint / "config.json")
        self.tokenizer_digest = _digest(checkpoint / "tokenizer.json")
        self.head_ref = self.root / "head.safetensors"
        self.head_bytes = self.head_weight.numel() * self.head_weight.element_size()
        self.bytes_used = 0
        self.tokens_used = 0
        self.published = 0
        self.copy_seconds = 0.0
        self.stop_reason: str | None = None

    def _sequences(self, batch: dict, rollout_data: dict, indices: list[int]) -> tuple[DraftSequence, ...]:
        sequences = []
        for local_index, rollout_index in enumerate(indices):
            sample_id = rollout_data["sample_indices"][rollout_index]
            if sample_id is None:
                raise ValueError("collect-only draft features require original sample indices")
            total = batch["total_lengths"][local_index]
            response = batch["response_lengths"][local_index]
            mask = tuple(int(value) for value in batch["loss_masks"][local_index].tolist())
            sequences.append(
                DraftSequence(
                    sample_id=sample_id,
                    group_id=rollout_data["group_indices"][rollout_index],
                    rollout_id=rollout_data["rollout_ids"][rollout_index],
                    token_ids=tuple(int(value) for value in batch["unconcat_tokens"][local_index].tolist()),
                    loss_mask=(0,) * (total - response - 1) + mask + (0,),
                    weight_versions=normalize_weight_versions(rollout_data["weight_versions"][rollout_index]),
                )
            )
        return tuple(sequences)

    def forward(
        self, model: torch.nn.Module, forward_kwargs: dict, batch: dict, rollout_data: dict, indices: list[int]
    ):
        if self.stop_reason is not None:
            return model(**forward_kwargs)
        sequences = self._sequences(batch, rollout_data, indices)
        selected = sum(sum(sequence.loss_mask) for sequence in sequences)
        if selected == 0:
            return model(**forward_kwargs)
        if self.published >= self.args.draft_feature_max_batches:
            self.stop_reason = "batch budget"
            return model(**forward_kwargs)
        if self.tokens_used + selected > self.args.draft_feature_max_tokens:
            self.stop_reason = "token budget"
            return model(**forward_kwargs)
        payload_bytes = selected * self.head_weight.shape[1] * self.head_weight.element_size()
        next_bytes = payload_bytes + (self.head_bytes if self.published == 0 else 0)
        if self.bytes_used + next_bytes > self.args.draft_feature_max_bytes:
            self.stop_reason = "byte budget"
            return model(**forward_kwargs)

        token_map = build_token_map(sequences, self.args.draft_feature_max_tokens - self.tokens_used)
        positions = torch.tensor(
            [token.packed_position for token in token_map.selected_tokens],
            device=self.head_weight.device,
        )
        captured: torch.Tensor | None = None

        def capture_head_input(_module, inputs):
            nonlocal captured
            if captured is not None:
                return
            hidden = inputs[0]
            if hidden.ndim == 3 and hidden.shape[1] == 1:
                hidden = hidden[:, 0]
            elif hidden.ndim != 2:
                raise ValueError("collect-only draft features require sequence-first batch size one")
            actual_bytes = selected * hidden.shape[-1] * hidden.element_size()
            if (
                self.bytes_used + actual_bytes + (self.head_bytes if self.published == 0 else 0)
                > self.args.draft_feature_max_bytes
            ):
                self.stop_reason = "byte budget"
                return
            # index_select owns storage, including when the source is already on CPU.
            captured = hidden.detach().index_select(0, positions)

        handle = self.output_layer.register_forward_pre_hook(capture_head_input)
        try:
            output = model(**forward_kwargs)
        finally:
            handle.remove()
        if captured is None:
            if self.stop_reason is not None:
                return output
            raise RuntimeError("target forward did not reach the LM head")

        copy_start = time.perf_counter()
        features = captured.cpu().contiguous()
        payload_bytes = features.numel() * features.element_size()
        next_bytes = payload_bytes + (self.head_bytes if self.published == 0 else 0)
        payload_ref = self.root / f"batch-{self.published:04d}.safetensors"
        manifest = DraftFeatureManifest(
            schema_version=1,
            feature_batch_id=f"{self.args.draft_feature_run_id}/round-{self.round_id}/batch-{self.published}",
            run_id=self.args.draft_feature_run_id,
            round_id=self.round_id,
            policy_source_version=self.source_version,
            head_source_version=self.source_version,
            model_tag="actor",
            model_config_digest=self.model_digest,
            tokenizer_digest=self.tokenizer_digest,
            capture_point="lm_head_input",
            dtype=str(features.dtype).removeprefix("torch."),
            token_map=token_map,
            head_snapshot_ref=self.head_ref.name,
            payload_ref=payload_ref.name,
            byte_count=payload_bytes,
            ready=True,
            hidden_size=features.shape[1],
        )
        manifest_ref = self.root / f"batch-{self.published:04d}.json"
        owned = [payload_ref, manifest_ref]
        if self.published == 0:
            owned.append(self.head_ref)
        committed = False
        try:
            if self.published == 0:
                head = self.head_weight.detach().to(device="cpu", copy=True).contiguous()
                save_file({"weight": head}, str(self.head_ref) + ".tmp")
                os.replace(str(self.head_ref) + ".tmp", self.head_ref)
            save_file({"features": features}, str(payload_ref) + ".tmp")
            os.replace(str(payload_ref) + ".tmp", payload_ref)
            Path(str(manifest_ref) + ".tmp").write_text(json.dumps(asdict(manifest), sort_keys=True))
            os.replace(str(manifest_ref) + ".tmp", manifest_ref)
            committed = True
        finally:
            for path in owned:
                Path(str(path) + ".tmp").unlink(missing_ok=True)
                if not committed:
                    path.unlink(missing_ok=True)
        self.copy_seconds += time.perf_counter() - copy_start
        self.bytes_used += next_bytes
        self.tokens_used += selected
        self.published += 1
        return output

    def finish(self) -> None:
        if self.published == 0:
            reason = self.stop_reason or "no selected target tokens"
            raise ValueError(f"collect-only draft features produced no batch: {reason}")
        summary = {
            "source_version": self.source_version,
            "batches": self.published,
            "tokens": self.tokens_used,
            "bytes": self.bytes_used,
            "copy_seconds": self.copy_seconds,
            "stop_reason": self.stop_reason,
        }
        (self.root / "summary.json").write_text(json.dumps(summary, sort_keys=True))
