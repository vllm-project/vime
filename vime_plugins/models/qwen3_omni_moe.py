"""Native Megatron model for Qwen3-Omni-MoE Thinker training.

Architecture (Thinker-only, Talker/Code2Wav are frozen and not trained):
  HF audio encoder  (Qwen3OmniMoeAudioEncoder,  replicated on first PP stage)
  HF vision encoder (Qwen3OmniMoeVisionEncoder, replicated on first PP stage)
  + Megatron GPTModel (MoE language model with M-RoPE, deepstack)

The forward pass:
  1. Computes text embeddings from `input_ids`.
  2. Runs the HF vision encoder on `pixel_values`+`image_grid_thw`
     (and `pixel_values_videos`+`video_grid_thw` if present), scatters the
     resulting vision embeddings into the combined embedding tensor at
     positions where `input_ids == image_token_id` / `video_token_id`.
  3. Runs the HF audio encoder on `input_features`+`feature_attention_mask`,
     scatters the resulting audio embeddings at positions where
     `input_ids == audio_token_id`.
  4. Computes 3D M-RoPE position IDs from the full input_ids + grid info
     (audio-aware, ported from Relax's get_rope_index).
  5. Forwards the combined embeddings + M-RoPE position IDs through the
     Megatron GPTModel (MoE) language model.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import torch
from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import deprecate_inference_params

from .qwen3_5_vl import Qwen3_5MultimodalRotaryEmbedding
from .qwen3_5_vl_utils import gather_packed_input_ids
from .qwen3_omni_transformer import Qwen3OmniTransformerBlock, split_deepstack_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# THD <-> batch-sequence helpers
# ---------------------------------------------------------------------------
def _thd_to_batch_seq(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Unpack THD-format [1, T, ...] to [bs, max_seq, ...] using cu_seqlens."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = seqlens.max().item()
    bs = len(cu_seqlens) - 1
    out = packed.new_zeros(bs, max_seq, *packed.shape[2:])
    for i, sl in enumerate(seqlens):
        out[i, :sl] = packed[0, cu_seqlens[i] : cu_seqlens[i] + sl]
    return out


def _batch_seq_to_thd(unpacked: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Pack [bs, max_seq, ...] back to THD [1, T, ...]."""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = cu_seqlens[-1].item()
    out = unpacked.new_zeros(1, total, *unpacked.shape[2:])
    for i, sl in enumerate(seqlens):
        out[0, cu_seqlens[i] : cu_seqlens[i] + sl] = unpacked[i, :sl]
    return out


def _gather_input_ids_from_cp(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the full packed sequence from Megatron's zigzag CP layout."""
    return gather_packed_input_ids(input_ids, cu_seqlens, parallel_state.get_context_parallel_group())


# ---------------------------------------------------------------------------
# Audio-aware M-RoPE position ID computation (ported from Relax)
# ---------------------------------------------------------------------------
def _get_feat_extract_output_lengths(input_lengths):
    """Computes the output length of the conv layers and the audio encoder."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


def _get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    audio_token_id: int,
    vision_start_token_id: int,
    audio_start_token_id: int,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    audio_seqlens: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    use_audio_in_video: bool = False,
    second_per_grids: torch.Tensor | None = None,
    position_id_per_seconds: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate RoPE position indices for multimodal inputs (audio+image+video).

    Ported from relax.models.qwen_omni.modeling_qwen3_omni.utils.get_rope_index.
    Returns position_ids of shape [3, batch, seq] for M-RoPE.
    """
    # Do NOT split video_grid_thw by repeat_interleave.
    # The Qwen3-Omni processor does NOT insert timestamp tokens between video
    # frames for pure video (use_audio_in_video=False). Input_ids have ONE
    # <vision_start> + (grid_t*grid_h*grid_w/merge^2) <video_pad> + <vision_end>.
    # Splitting grid_t into t=1 entries would under-count video tokens and break
    # M-RoPE vs vLLM (see vLLM get_mrope_input_positions).

    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=torch.float,
            device=input_ids.device,
        )
        image_index, video_index, audio_index = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]

            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids_i == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums

            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if (image_token_id in input_tokens or video_token_id in input_tokens) and (
                    remain_videos > 0 or remain_images > 0
                ):
                    ed_vision_start = input_tokens.index(vision_start_token_id, st)
                else:
                    ed_vision_start = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio_start = input_tokens.index(audio_start_token_id, st)
                else:
                    ed_audio_start = len(input_tokens) + 1
                min_ed = min(ed_vision_start, ed_audio_start)

                # ---------- text ----------
                text_len = min_ed - st
                if text_len > 0:
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    st_idx += text_len

                # ---------- BOS ----------
                if min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    bos_len, eos_len = 2, 2
                else:
                    bos_len, eos_len = 1, 1

                llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                st_idx += bos_len

                # Audio Only
                if min_ed == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_index += 1
                    remain_audios -= 1

                # Image Only
                elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == image_token_id:
                    t, h, w = (
                        image_grid_thw[image_index][0].item(),
                        image_grid_thw[image_index][1].item(),
                        image_grid_thw[image_index][2].item(),
                    )
                    t_index = (torch.arange(t) * 1 * position_id_per_seconds).float()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list.append(_llm_pos_ids + st_idx)

                    image_len = image_grid_thw[image_index].prod().item() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + image_len + eos_len)
                    image_index += 1
                    remain_images -= 1

                # Video Only
                elif min_ed == ed_vision_start and input_ids_i[ed_vision_start + 1] == video_token_id:
                    t, h, w = (
                        video_grid_thw[video_index][0].item(),
                        video_grid_thw[video_index][1].item(),
                        video_grid_thw[video_index][2].item(),
                    )
                    t_index = (
                        torch.arange(t) * second_per_grids[video_index].cpu().float() * position_id_per_seconds
                    ).float()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list.append(_llm_pos_ids + st_idx)

                    video_len = video_grid_thw[video_index].prod().item() // (spatial_merge_size**2)
                    st += int(text_len + bos_len + video_len + eos_len)
                    video_index += 1
                    remain_videos -= 1

                # Audio in Video
                elif min_ed == ed_vision_start and ed_vision_start + 1 == ed_audio_start:
                    audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_index])
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx

                    t, h, w = (
                        video_grid_thw[video_index][0].item(),
                        video_grid_thw[video_index][1].item(),
                        video_grid_thw[video_index][2].item(),
                    )
                    t_index = (
                        torch.arange(t) * second_per_grids[video_index].cpu().float() * position_id_per_seconds
                    ).float()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    h_index = (
                        torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten().float()
                    )
                    w_index = (
                        torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten().float()
                    )
                    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().float()
                    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
                    llm_pos_ids_list_temp = [_llm_pos_ids + st_idx]
                    video_llm_pos_ids = torch.cat(llm_pos_ids_list_temp, dim=1)

                    video_data_index, audio_data_index = 0, 0
                    while (
                        video_data_index < video_llm_pos_ids.shape[-1]
                        and audio_data_index < audio_llm_pos_ids.shape[-1]
                    ):
                        if video_llm_pos_ids[0][video_data_index] <= audio_llm_pos_ids[0][audio_data_index]:
                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_data_index + 1])
                            video_data_index += 1
                        else:
                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_data_index + 1])
                            audio_data_index += 1
                    if video_data_index < video_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(video_llm_pos_ids[:, video_data_index : video_llm_pos_ids.shape[-1]])
                    if audio_data_index < audio_llm_pos_ids.shape[-1]:
                        llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_data_index : audio_llm_pos_ids.shape[-1]])
                    video_len = video_grid_thw[video_index].prod().item() // (spatial_merge_size**2)

                    st += int(text_len + bos_len + audio_len + video_len + eos_len)
                    audio_index += 1
                    video_index += 1
                    remain_videos -= 1
                    remain_audios -= 1
                else:
                    raise RuntimeError("unexpected error in get_rope_index")

                # ---------- EOS ----------
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

            # tail text
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat([item.float() for item in llm_pos_ids_list], dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_i))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # fallback (pure text)
        if attention_mask is not None:
            position_ids = attention_mask.float().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# GPTModel with DeepStack support (keeps MCore mrope, swaps decoder only)
# ---------------------------------------------------------------------------
class Qwen3OmniMultimodalRotaryEmbedding(Qwen3_5MultimodalRotaryEmbedding):
    """Qwen3 interleaved MRoPE with packed-sequence CP handling."""

    def __init__(self, *args, cp_group=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cp_group = cp_group
        self.is_thd_format = False

    def forward(self, position_ids, mrope_section, packed_seq_params=None, **kwargs):
        return super().forward(
            position_ids,
            mrope_section,
            packed_seq=self.is_thd_format,
            cp_group=self.cp_group,
        )


class Qwen3OmniMoeGPTModel(MCoreGPTModel):
    """Qwen3-Omni GPT model with DeepStack support.

    Inherits GPTModel to keep MCore's MultimodalRotaryEmbedding (proven for text
    training). Only replaces the decoder to add DeepStack
    injection at the first N decoder layers.
    """

    def __init__(
        self,
        config,
        transformer_layer_spec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: str = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor=None,
        mtp_block_spec=None,
        vp_stage=None,
        pg_collection=None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        # Match HF/vLLM's interleaved multimodal RoPE layout.
        # CRITICAL: MCore's MultimodalRotaryEmbedding uses NON-interleaved mrope
        # layout [T48,H40,W40,...] which diverges from vLLM/HF interleaved layout
        # [T24,H24,W24,...] when t!=h!=w (video). For text (t=h=w) both are identical.
        cp_group = None
        if pg_collection is not None and getattr(pg_collection, "cp", None) is not None:
            cp_group = pg_collection.cp
        else:
            from megatron.core import parallel_state

            cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        self.rotary_pos_emb = Qwen3OmniMultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=False,  # bridge asserts not interleaved; uses apply_interleaved_mrope internally
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=cp_group,
        )
        # Rebuild the decoder with DeepStack injection.
        self.decoder = Qwen3OmniTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs=None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask=None,
        # args for deepstack
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
    ):
        """Forward pass with DeepStack visual embedding injection."""
        inference_context = deprecate_inference_params(inference_context, inference_params)

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        result = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )
        return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Qwen3OmniMoeVLModel(MegatronModule):
    """Qwen3-Omni-MoE Thinker model for Megatron training.

    Wraps an HF audio encoder and an HF vision encoder (only on first PP stage)
    together with a standard Megatron Core GPTModel configured for M-RoPE
    (MoE language model).

    Thinker-only training: the Talker and Code2Wav modules are not loaded.
    The audio and vision encoders are frozen by default (RL only trains the
    language model).
    """

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        hf_audio_config,
        hf_vision_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection=None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = pg_collection

        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id
        self.audio_token_id = language_transformer_config.audio_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.spatial_merge_size = language_transformer_config.spatial_merge_size
        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds
        self.use_audio_in_video = getattr(language_transformer_config, "use_audio_in_video", False)

        self.share_embeddings_and_output_weights = False

        # Encoders -- only on the first pipeline stage
        self.audio_model = None
        self.vision_model = None
        if self.pre_process:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
                Qwen3OmniMoeAudioEncoder,
                Qwen3OmniMoeVisionEncoder,
            )

            self.audio_model = Qwen3OmniMoeAudioEncoder._from_config(hf_audio_config)
            self.vision_model = Qwen3OmniMoeVisionEncoder._from_config(hf_vision_config)
            # Freeze encoders -- not trained during RL
            self.audio_model.requires_grad_(False)
            self.audio_model.eval()
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()

            for parameter in (*self.audio_model.parameters(), *self.vision_model.parameters()):
                parameter.tensor_model_parallel = False
                parameter.partition_dim = -1
                parameter.partition_stride = 1
            if torch.cuda.is_available():
                # Keep encoder param dtype (often bf16 from HF config); only move device.
                _enc_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                _audio_dtype = next(self.audio_model.parameters()).dtype
                _vision_dtype = next(self.vision_model.parameters()).dtype
                self.audio_model = self.audio_model.to(device=_enc_device, dtype=_audio_dtype)
                self.vision_model = self.vision_model.to(device=_enc_device, dtype=_vision_dtype)

        # Language model -- Megatron GPT with M-RoPE + DeepStack support
        self.language_model = Qwen3OmniMoeGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    # -- helpers required by Megatron pipeline engine -----------------------

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    # -- encoder helpers ----------------------------------------------------

    def _get_vision_features(self, pixel_values, image_grid_thw):
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        with torch.no_grad():
            outputs = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output, outputs.deepstack_features
        return outputs

    def _get_audio_features(self, input_features, feature_lens):
        with torch.no_grad():
            outputs = self.audio_model(
                input_features.to(dtype=self.audio_model.dtype),
                feature_lens=feature_lens,
            )
        return outputs.last_hidden_state

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        # multimodal kwargs
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        video_second_per_grid: torch.Tensor = None,
        # audio kwargs
        input_features: torch.Tensor = None,
        feature_attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the Qwen3-Omni Thinker model.

        Args:
            input_ids: [batch, seq] or THD [1, T] text token ids.
            position_ids: optional, otherwise computed from input_ids.
            attention_mask: text attention mask.
            pixel_values: image pixel values (flat, [N_pix, C*P*P]).
            image_grid_thw: [num_images, 3] (T, H, W) per image.
            pixel_values_videos: video pixel values.
            video_grid_thw: [num_videos, 3] (T, H, W) per video.
            image_input_mask: optional precomputed image mask.
            video_second_per_grid: seconds per video grid (for MRoPE t_index).
            input_features: audio mel features [batch, channels, time].
            feature_attention_mask: audio attention mask [batch, time].
        """
        assert inference_params is None, "Inference not supported"

        # Extract cu_seqlens and CP info early
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        cp_size = parallel_state.get_context_parallel_world_size()

        # Audio feature lengths (computed once, used by both audio encoder and MRoPE)
        audio_feature_lengths = None
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        # Vision bookkeeping
        video_start_index = 0
        vision_grid_thw = None
        vision_data = None
        image_mask = None
        video_mask = None
        deepstack_feature_lists = None

        combined_embeddings = None
        visual_pos_masks = None

        if self.pre_process:
            # =========================
            # Vision (image / video)
            # =========================
            if image_grid_thw is not None or video_grid_thw is not None:
                if image_grid_thw is not None:
                    image_mask = image_input_mask
                    if image_mask is None:
                        image_mask = (input_ids == self.image_token_id).contiguous()
                    vision_grid_thw = image_grid_thw
                    vision_data = pixel_values
                    video_start_index = image_mask.sum().item()
                else:
                    video_start_index = 0

                if video_grid_thw is not None:
                    video_mask = (input_ids == self.video_token_id).contiguous()
                    if vision_grid_thw is not None:
                        vision_grid_thw = torch.cat([vision_grid_thw, video_grid_thw], dim=0)
                        vision_data = torch.cat([vision_data, pixel_values_videos], dim=0)
                    else:
                        vision_grid_thw = video_grid_thw
                        vision_data = pixel_values_videos

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                vision_embeds, deepstack_feature_lists = self._get_vision_features(vision_data, vision_grid_thw)
                vision_embeds = vision_embeds.to(dtype=self.language_model.embedding.word_embeddings.weight.dtype)

            # =========================
            # Text embeddings
            # =========================
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()  # [seq, batch, hidden]

            # =========================
            # Scatter vision embeds
            # =========================
            if vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                        f"{video_start_index}"
                    )

                # [seq, bs, h] -> [bs, seq, h] for masked scatter
                combined_embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                if image_embeds is not None:
                    combined_embeddings_bsh[image_mask] = image_embeds
                if video_embeds is not None:
                    combined_embeddings_bsh[video_mask] = video_embeds
                combined_embeddings = combined_embeddings_bsh.transpose(0, 1).contiguous()

                if image_embeds is not None and video_embeds is not None:
                    visual_pos_masks = image_mask | video_mask
                elif image_embeds is not None:
                    visual_pos_masks = image_mask
                elif video_embeds is not None:
                    visual_pos_masks = video_mask

            # =========================
            # Audio
            # =========================
            if input_features is not None:
                audio_mask = (input_ids == self.audio_token_id).contiguous()
                # Flatten input_features using feature_attention_mask
                if feature_attention_mask is not None:
                    input_features_flat = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
                else:
                    input_features_flat = input_features

                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )

                audio_embeds = self._get_audio_features(input_features_flat, feature_lens)
                audio_embeds = audio_embeds.to(combined_embeddings.dtype)

                combined_embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings_bsh[audio_mask] = audio_embeds
                combined_embeddings = combined_embeddings_bsh.transpose(0, 1).contiguous()

            # Scatter to sequence-parallel region if needed
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        # =========================
        # Compute M-RoPE position IDs
        # =========================
        # position_ids must be available on ALL PP stages for rotary embeddings.
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            if self.pre_process:
                # Reconstruct full input_ids if CP > 1
                if cu_seqlens is not None:
                    if cp_size > 1:
                        full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    else:
                        full_input_ids = input_ids
                    input_ids_batch_seq = _thd_to_batch_seq(full_input_ids, cu_seqlens)
                else:
                    input_ids_batch_seq = input_ids

                # If no multimodal inputs at all, fall back to pure-text positions
                has_multimodal = (
                    (image_grid_thw is not None and image_grid_thw.numel() > 0)
                    or (video_grid_thw is not None and video_grid_thw.numel() > 0)
                    or audio_feature_lengths is not None
                )

                if has_multimodal:
                    pos_batch_seq, _ = _get_rope_index(
                        spatial_merge_size=self.spatial_merge_size,
                        image_token_id=self.image_token_id,
                        video_token_id=self.video_token_id,
                        audio_token_id=self.audio_token_id,
                        vision_start_token_id=self.vision_start_token_id,
                        audio_start_token_id=self.audio_start_token_id,
                        input_ids=input_ids_batch_seq,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        audio_seqlens=audio_feature_lengths,
                        attention_mask=None,
                        use_audio_in_video=self.use_audio_in_video,
                        second_per_grids=video_second_per_grid,
                        position_id_per_seconds=self.position_id_per_seconds,
                    )
                else:
                    # Pure text: standard 1D positions replicated across 3 dims
                    bs, seq_len = input_ids_batch_seq.shape
                    pos = torch.arange(seq_len, device=input_ids_batch_seq.device).unsqueeze(0).expand(bs, -1)
                    pos_batch_seq = torch.stack([pos, pos, pos], dim=0)  # [3, bs, seq]

                if cu_seqlens is not None:
                    pos_packed = _batch_seq_to_thd(pos_batch_seq.permute(1, 2, 0), cu_seqlens)
                    position_ids = pos_packed.permute(2, 0, 1).contiguous()  # [3, 1, T_global]
                else:
                    position_ids = pos_batch_seq  # [3, bs, seq]
            else:
                # Non-first PP stage: allocate buffer with correct shape
                if cu_seqlens is not None:
                    T = cu_seqlens[-1].item()
                    position_ids = torch.zeros(3, 1, T, dtype=torch.float, device=torch.cuda.current_device())
                else:
                    raise NotImplementedError(
                        "Non-THD position_ids broadcast not yet supported for non-first PP stages"
                    )

            # Broadcast position_ids from first to all PP stages
            if pp_size > 1:
                src = parallel_state.get_pipeline_model_parallel_first_rank()
                torch.distributed.broadcast(
                    position_ids,
                    src=src,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        # =========================
        # Split deepstack features for SP / CP
        # =========================
        if self.config.sequence_parallel and visual_pos_masks is not None and deepstack_feature_lists is not None:
            if self.pg_collection is not None:
                tp_size = self.pg_collection.tp.size()
                tp_rank = self.pg_collection.tp.rank()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()
                tp_rank = mpu.get_tensor_model_parallel_rank()
            visual_pos_masks, deepstack_feature_lists = split_deepstack_embeddings(
                visual_pos_masks,
                deepstack_feature_lists,
                tp_size=tp_size,
                tp_rank=tp_rank,
                sequence_parallel=True,
            )

        # =========================
        # Packed THD RoPE is sliced by attention, not by the embedding module.
        # =========================
        # Standard Qwen3-VL model sets is_thd_format=True dynamically (model.py:805,822)
        # when using packed sequences with CP. Otherwise the embedding module
        # slices emb along CP (because
        # packed_seq kwarg is swallowed by **kwargs and is_thd_format stays False),
        # producing freqs with T_global/cp_size entries. Then _apply_rotary_pos_emb_thd
        # CASE 2 (_get_thd_freqs_on_this_cp_rank) accesses out-of-bounds indices for
        # long sequences, producing a shorter freqs_packed that mismatches t.
        # Fix: set is_thd_format=True for packed (THD) sequences so CP slicing is
        # skipped here; _apply_rotary_pos_emb_thd handles CP per-sequence internally.
        if hasattr(self.language_model, "rotary_pos_emb") and hasattr(
            self.language_model.rotary_pos_emb, "is_thd_format"
        ):
            self.language_model.rotary_pos_emb.is_thd_format = cu_seqlens is not None

        # =========================
        # Language model forward
        # =========================
        # NOTE: visual_pos_masks and deepstack_visual_embeds are Qwen3-Omni-specific
        # DeepStack parameters. Standard Megatron GPTModel does not accept them; they
        # require custom decoder layers. Only pass when not None so text-only training
        # works with the standard GPTModel. Visual inputs need custom decoder support.
        language_model_kwargs = dict(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
        )
        if visual_pos_masks is not None:
            language_model_kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_feature_lists is not None:
            language_model_kwargs["deepstack_visual_embeds"] = deepstack_feature_lists
        if extra_block_kwargs:
            language_model_kwargs.update(extra_block_kwargs)

        output = self.language_model(**language_model_kwargs)

        return output


# ---------------------------------------------------------------------------
# Native model provider
# ---------------------------------------------------------------------------
def get_qwen3_omni_model_provider(args, config, vp_stage):
    """Return the native Qwen3-Omni Thinker provider."""

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    thinker_config = hf_config.thinker_config
    audio_config = deepcopy(thinker_config.audio_config)
    vision_config = deepcopy(thinker_config.vision_config)
    audio_config.torch_dtype = config.params_dtype
    vision_config.torch_dtype = config.params_dtype

    text_config = thinker_config.text_config
    rope_config = getattr(text_config, "rope_parameters", None) or getattr(text_config, "rope_scaling", None) or {}
    values = {
        "audio_start_token_id": thinker_config.audio_start_token_id,
        "audio_token_id": thinker_config.audio_token_id,
        "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
        "image_token_id": thinker_config.image_token_id,
        "language_max_sequence_length": args.max_position_embeddings,
        "mrope_section": list(rope_config.get("mrope_section", [24, 20, 20])),
        "position_id_per_seconds": thinker_config.position_id_per_seconds,
        "rotary_base": args.rotary_base,
        "rotary_percent": args.rotary_percent,
        "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
        "spatial_merge_size": vision_config.spatial_merge_size,
        "use_audio_in_video": getattr(thinker_config, "use_audio_in_video", False),
        "video_token_id": thinker_config.video_token_id,
        "vision_start_token_id": thinker_config.vision_start_token_id,
        "vocab_size": args.padded_vocab_size,
    }
    for name, value in values.items():
        setattr(config, name, value)

    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        fp8=False,
    )

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> Qwen3OmniMoeVLModel:
        return Qwen3OmniMoeVLModel(
            language_transformer_config=config,
            language_transformer_layer_spec=layer_spec,
            hf_audio_config=audio_config,
            hf_vision_config=vision_config,
            pre_process=pre_process,
            post_process=post_process,
        )

    return model_provider
