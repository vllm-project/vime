"""Hidden state capture for DSpark draft model training.

Adapted from NeMo RL's ``nemo_rl/models/megatron/draft/hidden_capture.py``.

During the policy forward pass, forward hooks capture:
    1. Embedding output (input_embeds) — from the policy's embedding layer
    2. Hidden states at ``target_layer_ids`` — concatenated for DSpark's fc projection
    3. Last layer hidden states — for L_tv / L_conf loss computation

All captured states are gathered to the last pipeline stage (where the draft
model runs). PP=1 skips the send/recv path entirely.

Key difference from NeMo RL Eagle3:
    - Eagle3 captures ``hidden_states`` (concatenated aux layers) + ``inputs_embeds``
    - DSpark additionally captures ``target_last_hidden_states`` (policy's final layer)
      because L_tv and L_conf require the policy's prediction at each draft position.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


# Dtype encoding for send/recv (matches NeMo RL)
_DTYPE_TO_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


@dataclass
class CapturedStates:
    """Container for hidden states captured from the policy model.

    Attributes:
        target_hidden_states: [seq_len, bsz, num_target_layers * hidden]
            Concatenated hidden states from policy's target_layer_ids.
        inputs_embeds: [seq_len, bsz, hidden] embedding output (not used by
            DSpark training forward, but captured for potential debugging).
        target_last_hidden_states: [seq_len, bsz, hidden] policy's final layer
            hidden states. Used for L_tv / L_conf computation.
    """

    target_hidden_states: Tensor | None = None
    inputs_embeds: Tensor | None = None
    target_last_hidden_states: Tensor | None = None


class HiddenStateCapture:
    """Capture policy embeddings, aux-layer hidden states, and last-layer hidden states.

    This class registers forward hooks on the policy model's embedding layer,
    target layers (specified by ``target_layer_ids``), and the last decoder
    layer. After the policy forward pass, ``get_captured_states()`` returns
    the gathered tensors.

    For PP>1, hidden states from earlier stages are sent to the last stage
    via point-to-point send/recv. PP=1 skips this path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer_ids: tuple[int, ...],
        last_layer_idx: int | None = None,
    ):
        """Initialize the capture.

        Args:
            model: the policy model (will be unwrapped if DDP-wrapped)
            target_layer_ids: global layer indices to capture (0-indexed)
            last_layer_idx: global index of the last layer to capture for
                L_tv/L_conf. If None, uses ``max(target_layer_ids)`` or the
                model's num_layers - 1.
        """
        from megatron.core.utils import unwrap_model

        self.model = unwrap_model(model)
        self.target_layer_ids = tuple(int(i) for i in target_layer_ids)

        # Determine number of layers
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            self.num_layers = len(self.model.decoder.layers)
            self._decoder = self.model.decoder
        elif hasattr(self.model, "module"):
            inner = self.model.module
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                self.num_layers = len(inner.decoder.layers)
                self._decoder = inner.decoder
                self.model = inner
            else:
                raise RuntimeError("Cannot find decoder.layers in policy model. " f"Model type: {type(self.model)}")
        else:
            raise RuntimeError(f"Cannot find decoder.layers in policy model. Model type: {type(self.model)}")

        if last_layer_idx is None:
            self.last_layer_idx = self.num_layers - 1
        else:
            self.last_layer_idx = int(last_layer_idx)

        # PP info
        try:
            from megatron.core import parallel_state

            self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            self.is_first_stage = parallel_state.is_pipeline_first_stage()
            self.is_last_stage = parallel_state.is_pipeline_last_stage()
        except Exception:
            # No parallel state initialized (e.g., unit test)
            self.pp_size = 1
            self.pp_rank = 0
            self.is_first_stage = True
            self.is_last_stage = True

        # Map global layer idx -> local layer idx on this PP stage
        self._global_to_local: dict[int, int] = {}
        self._local_aux_indices: list[int] = []
        self._local_last_idx: int | None = None
        self._compute_local_layer_mapping()

        self._captured: dict[str, Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _compute_local_layer_mapping(self) -> None:
        """Map global layer indices to local indices on this PP stage."""
        for local_idx, layer in enumerate(self._decoder.layers):
            # Megatron layers have layer_number (1-indexed)
            global_idx = int(getattr(layer, "layer_number", local_idx + 1)) - 1
            if global_idx in self.target_layer_ids:
                self._global_to_local[global_idx] = local_idx
                self._local_aux_indices.append(local_idx)
            if global_idx == self.last_layer_idx:
                self._local_last_idx = local_idx

    def _make_layer_output_hook(self, key: str):
        def hook(_module, _args, output):
            # Megatron decoder layer output is typically (hidden_states, ...)
            hidden_states = output[0] if isinstance(output, tuple) else output
            if hidden_states is None:
                return
            self._captured[key] = hidden_states.detach().clone()

        return hook

    def _make_embedding_hook(self):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            self._captured["embeds"] = output.detach().clone()

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on embedding, target layers, and last layer."""
        self.clear_hooks()
        self._captured.clear()

        # Embedding hook (only on first PP stage)
        if self.is_first_stage:
            embedding = getattr(self.model, "embedding", None)
            if embedding is not None:
                self._hooks.append(embedding.register_forward_hook(self._make_embedding_hook()))

        # Target layer hooks
        for global_idx in self.target_layer_ids:
            local_idx = self._global_to_local.get(global_idx)
            if local_idx is not None:
                layer = self._decoder.layers[local_idx]
                self._hooks.append(
                    layer.register_forward_hook(self._make_layer_output_hook(f"target_layer_{global_idx}"))
                )

        # Last layer hook
        if self._local_last_idx is not None:
            layer = self._decoder.layers[self._local_last_idx]
            self._hooks.append(layer.register_forward_hook(self._make_layer_output_hook("last_layer")))

    def clear_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @contextmanager
    def capture_context(self):
        """Context manager that registers hooks on enter and clears on exit."""
        try:
            self.register_hooks()
            yield self
        finally:
            self.clear_hooks()

    def _assemble_local_states(self) -> CapturedStates:
        """Assemble captured states when PP=1 (all layers on one stage)."""
        embeds = self._captured.get("embeds")

        # Concatenate target layer hidden states in target_layer_ids order
        hidden_chunks = []
        for global_idx in self.target_layer_ids:
            tensor = self._captured.get(f"target_layer_{global_idx}")
            if tensor is not None:
                hidden_chunks.append(tensor)

        target_hidden = torch.cat(hidden_chunks, dim=-1) if hidden_chunks else None

        last_hidden = self._captured.get("last_layer")

        return CapturedStates(
            target_hidden_states=target_hidden,
            inputs_embeds=embeds,
            target_last_hidden_states=last_hidden,
        )

    @staticmethod
    def _send_tensor(tensor: Tensor, dst_rank: int, group) -> None:
        """Send a tensor with metadata (shape + dtype)."""
        dtype_code = _DTYPE_TO_CODE.get(tensor.dtype)
        if dtype_code is None:
            raise ValueError(f"Unsupported tensor dtype for send/recv: {tensor.dtype}")
        metadata = torch.tensor(
            [tensor.shape[0], tensor.shape[1], tensor.shape[2], dtype_code],
            dtype=torch.int64,
            device=tensor.device,
        )
        dist.send(metadata, dst=dst_rank, group=group)
        dist.send(tensor.contiguous(), dst=dst_rank, group=group)

    @staticmethod
    def _recv_tensor(src_rank: int, group, device: torch.device) -> Tensor:
        """Receive a tensor with metadata."""
        metadata = torch.empty(4, dtype=torch.int64, device=device)
        dist.recv(metadata, src=src_rank, group=group)
        s0, s1, s2, dtype_code = [int(x) for x in metadata.tolist()]
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(f"Unsupported dtype code in recv: {dtype_code}")
        received = torch.empty(s0, s1, s2, dtype=dtype, device=device)
        dist.recv(received, src=src_rank, group=group)
        return received

    def _gather_distributed(self) -> CapturedStates:
        """Gather captured states from all PP stages to the last stage.

        Each target layer's owner rank sends its captured hidden states to the
        last PP stage. The embedding is sent from the first stage to the last.
        """
        from megatron.core import parallel_state

        pp_group = parallel_state.get_pipeline_model_parallel_group()
        last_rank = self.pp_size - 1
        recv_device = torch.device("cuda", torch.cuda.current_device())

        # If this stage has no captured tensors and is not the last stage, return empty
        if not self._captured and not self.is_last_stage:
            return CapturedStates()

        gathered_target: dict[int, Tensor] = {}
        gathered_last: Tensor | None = None
        gathered_embeds: Tensor | None = None

        # Gather target layer hidden states
        for global_idx in self.target_layer_ids:
            key = f"target_layer_{global_idx}"
            tensor = self._captured.get(key)
            if tensor is None:
                # This stage doesn't own this layer; last stage receives from owner
                if self.is_last_stage:
                    # Determine owner rank (simplified: assume even split)
                    layers_per_rank = max(1, self.num_layers // self.pp_size)
                    owner_rank = min(global_idx // layers_per_rank, self.pp_size - 1)
                    if owner_rank != self.pp_rank:
                        gathered_target[global_idx] = self._recv_tensor(
                            src_rank=owner_rank, group=pp_group, device=recv_device
                        )
                continue
            # This stage owns the layer
            if self.is_last_stage:
                gathered_target[global_idx] = tensor
            else:
                self._send_tensor(tensor, dst_rank=last_rank, group=pp_group)

        # Gather last layer hidden states
        last_tensor = self._captured.get("last_layer")
        if last_tensor is not None and not self.is_last_stage:
            self._send_tensor(last_tensor, dst_rank=last_rank, group=pp_group)
        elif self.is_last_stage and last_tensor is not None:
            gathered_last = last_tensor
        elif self.is_last_stage:
            # Receive from the stage that owns the last layer
            layers_per_rank = max(1, self.num_layers // self.pp_size)
            owner_rank = min(self.last_layer_idx // layers_per_rank, self.pp_size - 1)
            if owner_rank != self.pp_rank:
                gathered_last = self._recv_tensor(src_rank=owner_rank, group=pp_group, device=recv_device)

        # Gather embeddings
        if self.is_first_stage:
            embeds = self._captured.get("embeds")
            if embeds is not None:
                if self.is_last_stage:
                    gathered_embeds = embeds
                else:
                    self._send_tensor(embeds, dst_rank=last_rank, group=pp_group)
        elif self.is_last_stage:
            gathered_embeds = self._recv_tensor(src_rank=0, group=pp_group, device=recv_device)

        if not self.is_last_stage:
            return CapturedStates()

        # Concatenate target layers in order
        hidden_chunks = []
        for global_idx in self.target_layer_ids:
            tensor = gathered_target.get(global_idx)
            if tensor is not None:
                hidden_chunks.append(tensor)
        target_hidden = torch.cat(hidden_chunks, dim=-1) if hidden_chunks else None

        return CapturedStates(
            target_hidden_states=target_hidden,
            inputs_embeds=gathered_embeds,
            target_last_hidden_states=gathered_last,
        )

    def get_captured_states(self) -> CapturedStates:
        """Return captured states, gathering across PP stages if needed."""
        if self.pp_size == 1:
            return self._assemble_local_states()
        return self._gather_distributed()


def forward_with_dspark(model, forward_kwargs, batch, target_layer_ids):
    from megatron.core import tensor_parallel
    from megatron.core.utils import unwrap_model

    capture = HiddenStateCapture(model, target_layer_ids)
    with capture.capture_context():
        output_tensor = model(**forward_kwargs)

    captured = capture.get_captured_states()
    policy_model = unwrap_model(model)
    draft_model = getattr(policy_model, "draft_model", None)
    if draft_model is None or captured.target_hidden_states is None:
        return output_tensor, None, None

    def batch_first(hidden_states):
        if hidden_states is None:
            return None
        if policy_model.config.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False
            )
        return hidden_states.transpose(0, 1).contiguous()

    outputs = draft_model(
        input_ids=batch["tokens"],
        target_hidden_states=batch_first(captured.target_hidden_states),
        loss_mask=batch["full_loss_masks"],
        target_last_hidden_states=batch_first(captured.target_last_hidden_states),
    )
    return output_tensor, outputs, draft_model.config
