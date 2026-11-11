import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import divide, init_method_normal
from transformers.activations import ACT2FN

from vime.utils import accelerator

# Optional GDN conv/norm modules, selected by --qwen-gdn-backend ('fla_npu' or 'fla').
try:
    from .fla_npu_gdn import QwenStyleRMSNormGated as _FlaNpuRMSNormGated
    from .fla_npu_gdn import ShortConvolution as _FlaNpuShortConvolution
except ImportError:
    _FlaNpuRMSNormGated = None
    _FlaNpuShortConvolution = None

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except ImportError:
    FusedRMSNormGated = None
    ShortConvolution = None

from .hf_attention import HuggingfaceAttention, _load_hf_config
from .qwen_gdn_backend import get_chunk_gated_delta_rule


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    def __init__(self, config, layer_idx: int, args=None, megatron_config=None):
        super().__init__()
        self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla_npu")
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule(self.gdn_backend)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # --- TP-sharding (tensor parallel) ---
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.num_v_heads_local_tp = divide(self.num_v_heads, self.tp_size)
        self.num_k_heads_local_tp = divide(self.num_k_heads, self.tp_size)
        self.key_dim_local_tp = self.head_k_dim * self.num_k_heads_local_tp
        self.value_dim_local_tp = self.head_v_dim * self.num_v_heads_local_tp

        # QKV conv dimension (TP-local)
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_dim_local_tp = self.key_dim_local_tp * 2 + self.value_dim_local_tp

        # Head-first conv path for fla_npu; conv_head_num is the TP-local head count.
        conv_head_num = 0
        if self.gdn_backend == "fla_npu" and self.head_k_dim == self.head_v_dim and self.head_k_dim % 16 == 0:
            conv_head_num = 2 * self.num_k_heads_local_tp + self.num_v_heads_local_tp
        self.conv_head_num = conv_head_num
        if self.gdn_backend == "fla_npu":
            self.conv1d = _FlaNpuShortConvolution(
                hidden_size=self.conv_dim_local_tp,
                bias=False,
                kernel_size=self.conv_kernel_size,
                head_num=self.conv_head_num,
            )
        else:
            if ShortConvolution is None:
                raise ImportError(
                    "The 'fla'/'flashqla' GDN backends require flash-linear-attention "
                    "for the conv/norm modules. Install it with: "
                    "pip install --no-deps einops 'fla-core==0.5.2' 'flash-linear-attention==0.5.2'"
                )
            self.conv1d = ShortConvolution(
                hidden_size=self.conv_dim_local_tp,
                bias=False,
                kernel_size=self.conv_kernel_size,
            )
        # conv1d.weight is TP-sharded along dim 0; its q/k/v sub-matrices
        # are each sharded by head independently.
        self._conv1d_split_sizes_full = [
            self.key_dim,
            self.key_dim,
            self.value_dim,  # q, k, v
        ]
        self.conv1d.weight.tensor_model_parallel = True
        self.conv1d.weight.partition_dim = 0
        self.conv1d.weight.partition_stride = 1
        self.conv1d.weight._conv1d_split_sizes_full = self._conv1d_split_sizes_full

        # --- Input projection (merged qkv+z+b+a CPL; the converters cat/split
        # the four HF weights and TP shards each sub-matrix by head) ---
        self._in_proj_split_sizes_full = [
            self.key_dim,
            self.key_dim,
            self.value_dim,  # qkv: q, k, v
            self.value_dim,  # z
            self.num_v_heads,  # b
            self.num_v_heads,  # a
        ]
        init_method = getattr(megatron_config, "init_method", init_method_normal(0.02))
        output_init_method = getattr(megatron_config, "output_layer_init_method", init_method_normal(0.02))

        self.in_proj = ColumnParallelLinear(
            self.hidden_size,
            sum(self._in_proj_split_sizes_full),
            config=megatron_config,
            init_method=init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            is_expert=False,
        )
        self.in_proj.sequence_parallel = False
        self.in_proj.weight._in_proj_split_sizes_full = self._in_proj_split_sizes_full

        # time step projection (TP-local)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads_local_tp))
        self.dt_bias.tensor_model_parallel = True
        self.dt_bias.partition_dim = 0
        self.dt_bias.partition_stride = 1

        A = torch.empty(self.num_v_heads_local_tp).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log.tensor_model_parallel = True
        self.A_log.partition_dim = 0
        self.A_log.partition_stride = 1

        if self.gdn_backend == "fla_npu":
            self.norm = _FlaNpuRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=accelerator.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )
        else:
            self.norm = FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
                device=accelerator.current_device(),
                dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
            )

        # --- Output projection (RowParallelLinear) ---
        self.out_proj = RowParallelLinear(
            self.value_dim,  # full input dim; RPL handles the split
            self.hidden_size,
            config=megatron_config,
            init_method=output_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
        )
        self.out_proj.sequence_parallel = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # Single merged projection: split TP-local output back into qkv,z,b,a.
        merged, _ = self.in_proj(hidden_states)
        qkv_dim = self.key_dim_local_tp * 2 + self.value_dim_local_tp
        z_dim = self.value_dim_local_tp
        ba_dim = self.num_v_heads_local_tp * 2
        mixed_qkv, z, ba = merged.split([qkv_dim, z_dim, ba_dim], dim=-1)
        b, a = ba.chunk(2, dim=-1)

        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # Convolution on the TP-local QKV; head-first output when conv_head_num > 0.
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )

        if self.conv_head_num > 0:
            # Head-first output: leading 2*nk heads are Q+K, trailing nv are V.
            num_k_heads = self.num_k_heads_local_tp
            num_v_heads = self.num_v_heads_local_tp
            q_heads, k_heads, v_heads = mixed_qkv.split(
                (num_k_heads, num_k_heads, num_v_heads),
                dim=1,
            )
            # [B, T, H, D//H] for the downstream [B, T, H, D] contract.
            query = q_heads.transpose(1, 2).contiguous()
            key = k_heads.transpose(1, 2).contiguous()
            value = v_heads.transpose(1, 2).contiguous()
        else:
            # Flat output: split along the last dim, matching HF layout.
            query, key, value = torch.split(
                mixed_qkv,
                [self.key_dim_local_tp, self.key_dim_local_tp, self.value_dim_local_tp],
                dim=-1,
            )
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        # GQA: repeat_interleave query/key to match value head count
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if self.gdn_backend == "flashqla":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            g = g.contiguous()
            beta = beta.contiguous()

        if self.gdn_backend == "fla_npu":
            # fla_npu expects head-first q/k/v: [B, H, T, D].
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # Qwen3.5 is a VLM model with nested text_config
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"

        self.linear_attn = Qwen3_5GatedDeltaNet(self.hf_config, self.hf_layer_idx, args=args, megatron_config=config)

        # CPL backward already all-reduces grad_input, so every rank holds the
        # same grad_input and the sequence-parallel split can be skipped.
        self._linear_attn_tp_sharded = False

        # Use a simple RMSNorm
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
