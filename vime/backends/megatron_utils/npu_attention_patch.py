import torch_npu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from torch import Tensor

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def npu_dot_product_attention_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    attn_mask_type: AttnMaskType = None,
    attention_bias: Tensor = None,
    packed_seq_params: PackedSeqParams | None = None,
):
    assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

    if packed_seq_params is None:
        n_head = query.shape[2]
    else:
        n_head = query.shape[1]

    sparse_mode = getattr(self.config, "sparse_mode", 0)
    if attn_mask_type == AttnMaskType.no_mask:
        sparse_mode = 0

    scale = self.softmax_scale

    pre_tockens = getattr(self.config, "pre_tockens", 65536)
    next_tockens = getattr(self.config, "next_tockens", 0)

    if packed_seq_params is not None:
        if isinstance(packed_seq_params.cu_seqlens_q, list):
            actual_seq_qlen = packed_seq_params.cu_seqlens_q
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv
        else:
            actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
        shape_order = "TND"
    else:
        actual_seq_qlen = None
        actual_seq_kvlen = None
        if rearrange is not None:
            query, key, value = [rearrange(x, "s b h d -> s b (h d)") for x in [query, key, value]]
        else:
            query = query.reshape(query.shape[0], query.shape[1], -1)
            key = key.reshape(key.shape[0], key.shape[1], -1)
            value = value.reshape(value.shape[0], value.shape[1], -1)
        shape_order = "SBH"

    output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        n_head,
        shape_order,
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=scale,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        keep_prob=1 - self.attention_dropout.p,
        inner_precise=0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )[0]

    return output


from megatron.core.transformer.dot_product_attention import DotProductAttention

DotProductAttention.forward = npu_dot_product_attention_forward
print(
    "[NPU PATCH] DotProductAttention.forward replaced with npu_fusion_attention "
    "BEFORE megatron_adaptor import",
    flush=True,
)
