"""Native Qwen3-VL: Megatron language model and a replicated HF vision tower."""

from __future__ import annotations

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig

from .qwen3_5_vl_utils import build_packed_mrope_position_ids
from .qwen3_omni_moe import Qwen3OmniMoeGPTModel
from .qwen3_omni_transformer import split_deepstack_embeddings


def _load_vision_model(hf_config, config):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    device = (
        torch.device("cpu") if config.use_cpu_initialization else torch.device("cuda", torch.cuda.current_device())
    )
    with device:
        vision_model = Qwen3VLVisionModel._from_config(hf_config.vision_config, attn_implementation="sdpa")
    vision_model.to(dtype=config.params_dtype)
    if config.recompute_granularity == "full":
        vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for parameter in vision_model.parameters():
        parameter.tensor_model_parallel = False
        parameter.partition_dim = -1
        parameter.partition_stride = 1
    return vision_model


class Qwen3VLModel(MegatronModule):
    def __init__(self, config, language_layer_spec, hf_config, args, *, pre_process, post_process, vp_stage):
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = hf_config.image_token_id
        self.video_token_id = hf_config.video_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.spatial_merge_size = hf_config.vision_config.spatial_merge_size

        text_config = hf_config.text_config
        rope = getattr(text_config, "rope_parameters", None) or text_config.rope_scaling
        config.mrope_section = list(rope["mrope_section"])
        config.position_embedding_type = "mrope"
        config.rotary_base = rope.get("rope_theta", getattr(text_config, "rope_theta", args.rotary_base))
        config.apply_rope_fusion = False
        # The main Omni GPT class is also usable with a dense Qwen layer spec:
        # its additions are interleaved MRoPE and checkpoint-aware DeepStack.
        self.language_model = Qwen3OmniMoeGPTModel(
            config=config,
            transformer_layer_spec=language_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type="mrope",
            rotary_percent=args.rotary_percent,
            rotary_base=config.rotary_base,
            scatter_embedding_sequence_parallel=False,
            vp_stage=vp_stage,
        )
        self.model = torch.nn.Module()
        self.model.visual = _load_vision_model(hf_config, config) if pre_process else None
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    @property
    def decoder(self):
        return self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def _inject_vision_embeddings(self, input_ids, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw):
        embeddings = self.language_model.embedding(input_ids=input_ids, position_ids=None).transpose(0, 1).clone()
        visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        positions, deepstack = [], []
        for values, grids, token_id in (
            (pixel_values, image_grid_thw, self.image_token_id),
            (pixel_values_videos, video_grid_thw, self.video_token_id),
        ):
            mask = input_ids == token_id
            if values is None:
                if grids is not None or mask.any():
                    raise ValueError("Qwen3-VL vision tokens/grids require matching pixel values")
                continue
            if grids is None:
                raise ValueError("Qwen3-VL pixel values require matching grid_thw")
            output = self.model.visual(values.to(dtype=self.model.visual.dtype), grid_thw=grids)
            if hasattr(output, "pooler_output"):
                features, layer_features = output.pooler_output, output.deepstack_features
            else:
                features, layer_features = output
            if mask.sum().item() != features.shape[0]:
                raise ValueError("Qwen3-VL token/features count mismatch")
            embeddings[mask] = features.to(embeddings)
            visual_mask |= mask
            positions.append(mask.flatten().nonzero(as_tuple=False).flatten())
            deepstack.append(layer_features)

        deepstack_features = None
        if deepstack:
            # Images and videos are encoded separately but can interleave in a sample.
            order = torch.cat(positions).argsort()
            deepstack_features = [
                torch.cat(features)[order].to(embeddings) for features in zip(*deepstack, strict=True)
            ]
        embeddings = embeddings.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings).contiguous()
            if deepstack_features is not None:
                # Unlike Omni's frozen tower, this tower remains trainable. Each
                # SP rank uses only part of DeepStack; sum its feature gradients
                # before backpropagating through the replicated vision tower.
                deepstack_features = [
                    tensor_parallel.copy_to_tensor_model_parallel_region(features) for features in deepstack_features
                ]
                visual_mask, deepstack_features = split_deepstack_embeddings(
                    visual_mask,
                    deepstack_features,
                    tp_size=mpu.get_tensor_model_parallel_world_size(),
                    tp_rank=mpu.get_tensor_model_parallel_rank(),
                    sequence_parallel=True,
                )
        return embeddings, visual_mask if deepstack_features is not None else None, deepstack_features

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=None,
        loss_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("Qwen3-VL native training requires THD packed sequences")
        if position_ids is None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
            if cu_seqlens is None:
                cu_seqlens = packed_seq_params.cu_seqlens_q
            position_ids = build_packed_mrope_position_ids(
                input_ids,
                cu_seqlens,
                image_grid_thw,
                video_grid_thw,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                spatial_merge_size=self.spatial_merge_size,
            )
        embeddings, visual_mask, deepstack = self._inject_vision_embeddings(
            input_ids, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
        )
        self.language_model.rotary_pos_emb.is_thd_format = True
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=embeddings,
            labels=labels,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=deepstack,
            **kwargs,
        )


def get_qwen3_vl_model_provider(args, config, vp_stage):
    """Use main's --spec provider interface without adding a Bridge branch."""
    if config.pipeline_model_parallel_size != 1 or config.context_parallel_size != 1:
        raise ValueError("Qwen3-VL native training currently supports PP=1 and CP=1")
    if args.mtp_num_layers:
        raise ValueError("Qwen3-VL native MTP is not supported")
    if args.transformer_impl != "transformer_engine":
        raise ValueError("Qwen3-VL native training requires the TE/MindSpeed layer spec")
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if hf_config.model_type != "qwen3_vl":
        raise ValueError(f"{args.hf_checkpoint} is not a Qwen3-VL checkpoint")
    layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)

    def model_provider(pre_process=True, post_process=True, vp_stage=None):
        return Qwen3VLModel(
            config, layer_spec, hf_config, args, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

    return model_provider
