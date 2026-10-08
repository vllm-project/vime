from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "cuda")
    pytest.importorskip("megatron.core")
    from vime_plugins.models import qwen3_vl

    return qwen3_vl


@pytest.fixture
def hf_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        image_token_id=10,
        video_token_id=20,
        vision_start_token_id=30,
        text_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 0], "mrope_interleaved": True},
            "rope_theta": 5000000,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "depth": 2,
            "num_heads": 2,
            "out_hidden_size": 8,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 4,
            "deepstack_visual_indexes": [0],
        },
    )


def test_vision_native_load_export_and_backward(native, hf_config, tmp_path, monkeypatch):
    from vime.backends.megatron_utils.hf_to_megatron.common import load_model_hf_weights
    from vime.backends.megatron_utils.hf_to_megatron.qwen3_vl import qwen3_vl_hf_tensor
    from vime.backends.megatron_utils.megatron_to_hf import convert_to_hf
    from vime.backends.megatron_utils.update_weight import common

    config = SimpleNamespace(use_cpu_initialization=True, params_dtype=torch.float32, recompute_granularity="full")
    vision = native._load_vision_model(hf_config, config)
    tensors = {"model.visual." + name: param.detach().clone() for name, param in vision.named_parameters()}
    save_file(tensors, tmp_path / "model.safetensors")
    with torch.no_grad():
        for parameter in vision.parameters():
            parameter.zero_()
    named = [("module.module.model.visual." + name, param) for name, param in vision.named_parameters()]
    monkeypatch.setattr(common, "named_params_and_buffers", lambda args, model: iter(named))
    load_model_hf_weights(SimpleNamespace(), [vision], tmp_path, hf_config, qwen3_vl_hf_tensor)

    for name, parameter in named:
        [(hf_name, exported)] = convert_to_hf(None, "qwen3_vl", name, parameter)
        assert torch.equal(exported, tensors[hf_name])
        assert parameter.requires_grad
        assert not parameter.tensor_model_parallel
        assert parameter.partition_dim == -1
    output = vision(torch.randn(4, 24), grid_thw=torch.tensor([[1, 2, 2]]))
    features, deepstack = (
        (output.pooler_output, output.deepstack_features) if hasattr(output, "pooler_output") else output
    )
    (features.square().sum() + deepstack[0].square().sum()).backward()
    assert vision.patch_embed.proj.weight.grad.abs().sum() > 0
    assert vision.deepstack_merger_list[0].linear_fc2.weight.grad.abs().sum() > 0


def _injection_model(native, *, sequence_parallel=False):
    class Embedding:
        def __call__(self, input_ids, position_ids):
            return input_ids.T[..., None].float().expand(-1, -1, 2).clone()

    class Vision:
        dtype = torch.float32

        def __call__(self, values, grid_thw):
            return SimpleNamespace(pooler_output=values, deepstack_features=[values * 2, values * 3])

    return SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=sequence_parallel),
        language_model=SimpleNamespace(embedding=Embedding()),
        model=SimpleNamespace(visual=Vision()),
        image_token_id=10,
        video_token_id=20,
    )


def test_vision_and_deepstack_follow_token_order_and_keep_gradients(native):
    model = _injection_model(native)
    image = torch.tensor([[100.0, 101.0]], requires_grad=True)
    video = torch.tensor([[200.0, 201.0]], requires_grad=True)
    embeddings, mask, deepstack = native.Qwen3VLModel._inject_vision_embeddings(
        model, torch.tensor([[7, 20, 8, 10]]), image, video, torch.tensor([[1, 2, 2]]), torch.tensor([[1, 2, 2]])
    )
    expected = torch.cat((video, image))
    assert torch.equal(embeddings[:, 0][mask[0]], expected)
    assert torch.equal(deepstack[0], expected * 2)
    (embeddings.sum() + sum(t.sum() for t in deepstack)).backward()
    assert torch.equal(image.grad, torch.full_like(image, 6))
    assert torch.equal(video.grad, torch.full_like(video, 6))


def test_deepstack_sp_routes_gradients_through_tp_copy(native, monkeypatch):
    copied = []
    monkeypatch.setattr(
        native.tensor_parallel, "copy_to_tensor_model_parallel_region", lambda value: copied.append(value) or value
    )
    monkeypatch.setattr(native.tensor_parallel, "scatter_to_sequence_parallel_region", lambda value: value.chunk(4)[1])
    monkeypatch.setattr(native.mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(native.mpu, "get_tensor_model_parallel_rank", lambda: 1)
    model = _injection_model(native, sequence_parallel=True)
    image = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _, mask, deepstack = native.Qwen3VLModel._inject_vision_embeddings(
        model, torch.tensor([[10, 7, 10, 7, 7, 7, 7, 7]]), image, None, torch.tensor([[1, 2, 4]]), None
    )
    assert len(copied) == 2
    assert mask.tolist() == [[True, False]]
    assert torch.equal(deepstack[0], image[1:] * 2)


def test_missing_vision_is_not_silently_treated_as_text(native):
    with pytest.raises(ValueError, match="pixel values"):
        native.Qwen3VLModel._inject_vision_embeddings(
            _injection_model(native), torch.tensor([[10]]), None, None, None, None
        )


def _vision_tp_worker(rank, rendezvous):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    from vime_plugins.models import qwen3_vl as native

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=45))
    try:
        copy = native.tensor_parallel.copy_to_tensor_model_parallel_region
        scatter = native.tensor_parallel.scatter_to_sequence_parallel_region
        with (
            patch.object(torch.cuda, "current_device", lambda: torch.device("cpu")),
            patch.object(
                native.tensor_parallel,
                "copy_to_tensor_model_parallel_region",
                lambda x: copy(x, group=dist.group.WORLD),
            ),
            patch.object(
                native.tensor_parallel,
                "scatter_to_sequence_parallel_region",
                lambda x: scatter(x, group=dist.group.WORLD),
            ),
            patch.object(native.mpu, "get_tensor_model_parallel_world_size", lambda: 4),
            patch.object(native.mpu, "get_tensor_model_parallel_rank", lambda: rank),
        ):
            image = torch.arange(8, dtype=torch.float32).view(4, 2).requires_grad_()
            embeddings, _, deepstack = native.Qwen3VLModel._inject_vision_embeddings(
                _injection_model(native, sequence_parallel=True),
                torch.tensor([[10, 7, 10, 7, 10, 7, 10, 7]]),
                image,
                None,
                torch.tensor([[1, 4, 4]]),
                None,
            )
            (embeddings.sum() + sum(t.sum() for t in deepstack)).backward()
            # All four replicas receive the same complete gradient, including
            # image features consumed by other SP ranks (1 + 2 + 3 = 6).
            torch.testing.assert_close(image.grad, torch.full_like(image, 6))
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
def test_trainable_vision_tp4_gradients_on_cpu(native, tmp_path):
    torch.multiprocessing.spawn(_vision_tp_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=4)


def test_deepstack_gradients_survive_main_recompute(native, monkeypatch):
    from torch.utils.checkpoint import checkpoint

    from vime_plugins.models.qwen3_omni_transformer import Qwen3OmniTransformerBlock

    class Layer(torch.nn.Module):
        def __init__(self, number):
            super().__init__()
            self.layer_number = number

        def forward(self, hidden_states, **kwargs):
            return hidden_states * 2, None

    block = Qwen3OmniTransformerBlock.__new__(Qwen3OmniTransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        fp8=False, distribute_saved_activations=False, recompute_method="uniform", recompute_num_layers=1
    )
    block.pre_process = True
    block.layers = torch.nn.ModuleList([Layer(1), Layer(2)])
    block.num_layers_per_pipeline_rank = 2
    monkeypatch.setattr(
        native.tensor_parallel, "checkpoint", lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True)
    )
    hidden = torch.ones(4, 1, 2, requires_grad=True)
    features = [torch.ones(1, 2, requires_grad=True), torch.ones(1, 2, requires_grad=True)]
    output = block._checkpointed_forward(
        hidden,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        visual_pos_masks=torch.tensor([[False, True, False, False]]),
        deepstack_visual_embeds=features,
    )
    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 4))
    torch.testing.assert_close(features[0].grad, torch.full_like(features[0], 2))
    torch.testing.assert_close(features[1].grad, torch.ones_like(features[1]))


def test_interleaved_mrope_matches_hf_for_packed_images(native, hf_config):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    from vime_plugins.models.qwen3_omni_moe import Qwen3OmniMultimodalRotaryEmbedding

    positions = native.build_packed_mrope_position_ids(
        torch.tensor([[30, 10, 10, 10, 10, 7, 30, 10, 8]]),
        [0, 6, 9],
        torch.tensor([[1, 4, 4], [1, 2, 2]]),
        None,
        image_token_id=10,
        video_token_id=20,
        vision_start_token_id=30,
        spatial_merge_size=2,
    )
    assert positions[:, 0, 6].tolist() == [0, 0, 0]
    hf_rope = Qwen3VLTextRotaryEmbedding(hf_config.text_config)
    # Exercise the real main RoPE arithmetic without allocating a CUDA buffer.
    rope = Qwen3OmniMultimodalRotaryEmbedding.__new__(Qwen3OmniMultimodalRotaryEmbedding)
    torch.nn.Module.__init__(rope)
    rope.inv_freq = hf_rope.inv_freq
    rope.seq_len_interpolation_factor = None
    rope.cp_group = None
    rope.is_thd_format = True
    freqs = rope(positions, [1, 1, 0])[:, 0, 0].unsqueeze(0)
    cos, sin = hf_rope(torch.zeros(1, 9, 4), positions)
    torch.testing.assert_close(freqs.cos(), cos)
    torch.testing.assert_close(freqs.sin(), sin)


@pytest.mark.parametrize(("pp", "cp", "mtp"), [(2, 1, None), (1, 2, None), (1, 1, 1)])
def test_provider_rejects_unimplemented_topologies(native, pp, cp, mtp):
    with pytest.raises(ValueError, match="PP=1 and CP=1|MTP"):
        native.get_qwen3_vl_model_provider(
            SimpleNamespace(mtp_num_layers=mtp),
            SimpleNamespace(pipeline_model_parallel_size=pp, context_parallel_size=cp),
            None,
        )


def test_provider_uses_main_dense_deepstack_gpt(native, hf_config, monkeypatch):
    calls = {}

    def gpt(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(share_embeddings_and_output_weights=False)

    monkeypatch.setattr(native, "Qwen3OmniMoeGPTModel", gpt)
    monkeypatch.setattr(native, "_load_vision_model", lambda *args: torch.nn.Linear(8, 8))
    monkeypatch.setattr(native.AutoConfig, "from_pretrained", lambda *args, **kwargs: hf_config)
    monkeypatch.setattr(
        native, "get_gpt_layer_with_transformer_engine_spec", lambda *, qk_layernorm: {"qk_layernorm": qk_layernorm}
    )
    args = SimpleNamespace(
        hf_checkpoint="unused",
        mtp_num_layers=None,
        transformer_impl="transformer_engine",
        normalization="RMSNorm",
        padded_vocab_size=32,
        max_position_embeddings=128,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        rotary_percent=1.0,
        rotary_base=1000000,
    )
    config = SimpleNamespace(pipeline_model_parallel_size=1, context_parallel_size=1, normalization="RMSNorm")
    model = native.get_qwen3_vl_model_provider(args, config, None)()
    assert calls["transformer_layer_spec"] == {"qk_layernorm": True}
    assert calls["config"] is config
    assert calls["config"].normalization == "RMSNorm"
    assert calls["position_embedding_type"] == "mrope"
    assert calls["rotary_base"] == 5000000
    assert calls["scatter_embedding_sequence_parallel"] is False
    assert config.mrope_section == [1, 1, 0]
    assert all(parameter.requires_grad for parameter in model.model.visual.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
