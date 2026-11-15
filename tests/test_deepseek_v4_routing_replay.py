"""Check the native hash-router replay callback used by DeepSeek-V4."""

from types import SimpleNamespace

import pytest
import torch

from vime.utils import routing_replay


@pytest.mark.parametrize("native_replay", [False, True])
def test_hash_router_topk_is_recorded_and_replayed(monkeypatch, native_replay):
    scores = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    hash_indices = torch.tensor([[2, 0], [0, 1]])
    recorded = []

    def hash_topk(values, topk, num_groups=None, group_topk=None):
        assert topk == 2 and num_groups is None and group_topk is None
        return values.gather(1, hash_indices), hash_indices

    class Replay:
        def record(self, indices):
            recorded.append(indices)

        def pop_forward(self):
            return recorded[0]

    class NativeReplay:
        def get_replay_topk(self, values, topk, num_groups, group_topk, compute_topk):
            return compute_topk(values, topk, num_groups, group_topk)

    monkeypatch.setattr(routing_replay, "ROUTING_REPLAY", Replay())
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    bridge = routing_replay._SlimeRouterReplay(NativeReplay() if native_replay else None)
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "record")
    _, selected = bridge.get_replay_topk(scores, 2, default_compute_topk=hash_topk)
    assert torch.equal(selected, hash_indices)
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    replayed_probs, replayed_indices = bridge.get_replay_topk(scores + 1, 2, default_compute_topk=hash_topk)
    assert torch.equal(replayed_indices, hash_indices)
    assert torch.equal(replayed_probs, (scores + 1).gather(1, hash_indices))


def test_ordered_topk_capture_preserves_hash_table(monkeypatch):
    scores = torch.tensor([[0.9, 0.1, 0.5]])
    hash_indices = torch.tensor([[2, 1]])
    monkeypatch.setattr(routing_replay, "ORDERED_TOPK_CAPTURE_ROUTER", SimpleNamespace(is_hash_layer=True))

    def hash_topk(values, topk, num_groups=None, group_topk=None):
        return values.gather(1, hash_indices), hash_indices

    _, selected = routing_replay._compute_topk_for_current_router(hash_topk, scores, 2)
    assert torch.equal(selected, hash_indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron hash router needs CUDA")
def test_megatron_hash_router_uses_vime_replay(monkeypatch):
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.router_replay import RouterReplay
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    class Replay:
        def record(self, indices):
            self.indices = indices.clone()

        def pop_forward(self):
            return (self.indices + 1) % 4

    RouterReplay.clear_global_router_replay_instances()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    try:
        config = TransformerConfig(
            num_layers=3,
            hidden_size=16,
            num_attention_heads=8,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            moe_router_dtype="fp32",
            moe_router_score_function="sqrtsoftplus",
            moe_num_hash_layers=2,
            hash_moe_vocab_size=128,
            moe_enable_routing_replay=True,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )
        router = TopKRouter(config=config, pg_collection=get_default_pg_collection()).cuda()
        router.set_layer_number(1)
        assert router.is_hash_layer
        routing_replay._install_router_replay_bridge(router)
        replay = Replay()
        monkeypatch.setattr(routing_replay, "ROUTING_REPLAY", replay)
        monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
        logits = torch.randn(4, 4, device="cuda")
        input_ids = torch.tensor([[0, 1, 2, 3]], device="cuda")

        monkeypatch.setenv("ROUTING_REPLAY_STAGE", "record")
        _, recorded = router._hash_routing(logits, input_ids, dense_output=True)
        assert torch.equal(recorded, router.tid2eid[input_ids.T.reshape(-1)].long())
        monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
        _, replayed = router._hash_routing(logits, input_ids, dense_output=True)
        assert torch.equal(replayed, (recorded + 1) % 4)
    finally:
        Utils.destroy_model_parallel()
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-backend routing needs CUDA")
def test_v4_router_matches_vllm_on_fixed_inputs():
    pytest.importorskip("vllm")
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias

    generator = torch.Generator(device="cuda").manual_seed(525)
    logits = torch.randn(8, 256, device="cuda", generator=generator)
    bias = torch.randn(256, device="cuda", generator=generator) * 0.01
    hidden = torch.empty(8, 16, device="cuda")
    train_probs, train_ids = topk_routing_with_score_function(
        logits,
        topk=6,
        score_function="sqrtsoftplus",
        expert_bias=bias,
        scaling_factor=1.5,
        dense_output=True,
    )
    rollout_probs, rollout_ids = fused_topk_bias(
        hidden,
        logits,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=bias,
        topk=6,
        renormalize=True,
        routed_scaling_factor=1.5,
    )
    train_order = train_ids.argsort(dim=-1)
    rollout_order = rollout_ids.argsort(dim=-1)
    assert torch.equal(train_ids.gather(1, train_order), rollout_ids.gather(1, rollout_order))
    torch.testing.assert_close(
        train_probs.gather(1, train_order),
        rollout_probs.gather(1, rollout_order),
        rtol=1e-4,
        atol=1e-5,
    )
    from megatron.core.transformer.moe.router import TopKRouter

    token_ids = torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0)
    table = (torch.arange(32, device="cuda", dtype=torch.int32)[:, None] + torch.arange(6, device="cuda")) % 256
    router = SimpleNamespace(
        score_function="sqrtsoftplus",
        config=SimpleNamespace(
            moe_router_force_load_balancing=False,
            moe_router_force_biased=None,
            moe_router_topk_scaling_factor=1.5,
        ),
        tid2eid=table,
        topk=6,
        router_replay=None,
    )
    train_probs, train_ids = TopKRouter._hash_routing(router, logits, token_ids, dense_output=True)
    rollout_probs, rollout_ids = fused_topk_bias(
        hidden,
        logits,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=None,
        topk=6,
        renormalize=True,
        input_tokens=token_ids.reshape(-1),
        hash_indices_table=table,
        routed_scaling_factor=1.5,
    )
    assert torch.equal(train_ids.int(), rollout_ids.int())
    torch.testing.assert_close(train_probs, rollout_probs, rtol=1e-4, atol=1e-5)
