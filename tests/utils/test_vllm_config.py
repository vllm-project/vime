"""Unit tests for VllmConfig multi-model parsing and get_model_url."""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_yaml(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.flush()
    return f.name


class TestVllmConfigUpdateWeights:
    def test_update_weights_defaults_to_none(self):
        """Models without explicit update_weights parse as None (resolved to True/False at runtime by VllmConfig.resolve based on hf_checkpoint match)."""
        from vime.backends.vllm_utils.vllm_config import VllmConfig

        path = _write_yaml(
            {
                "vllm": [
                    {
                        "name": "actor",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    }
                ]
            }
        )
        config = VllmConfig.from_yaml(path)
        assert len(config.models) == 1
        # Parsed default is None; VllmConfig.resolve() later infers True/False from
        # whether model_path matches args.hf_checkpoint.
        assert config.models[0].update_weights is None

    def test_update_weights_explicit_false(self):
        """Models with update_weights: false should be parsed correctly."""
        from vime.backends.vllm_utils.vllm_config import VllmConfig

        path = _write_yaml(
            {
                "vllm": [
                    {
                        "name": "actor",
                        "update_weights": True,
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "model_path": "/path/to/ref",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
                    },
                ]
            }
        )
        config = VllmConfig.from_yaml(path)
        assert len(config.models) == 2
        assert config.models[0].name == "actor"
        assert config.models[0].update_weights is True
        assert config.models[1].name == "ref"
        assert config.models[1].update_weights is False
        assert config.models[1].model_path == "/path/to/ref"

    def test_multi_model_total_gpus(self):
        """total_num_gpus should sum across all models."""
        from vime.backends.vllm_utils.vllm_config import VllmConfig

        path = _write_yaml(
            {
                "vllm": [
                    {
                        "name": "actor",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                ]
            }
        )
        config = VllmConfig.from_yaml(path)
        assert config.total_num_gpus == 12

    def test_config_allows_model_with_no_server_groups(self):
        """A model with no server groups can expose a router without local engines."""
        from vime.backends.vllm_utils.vllm_config import VllmConfig

        path = _write_yaml({"vllm": [{"name": "default", "server_groups": []}]})

        config = VllmConfig.from_yaml(path)

        assert len(config.models) == 1
        assert config.models[0].name == "default"
        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0


class TestZeroGpuRolloutConfig:
    def test_resolve_default_zero_gpu_config_has_no_server_groups(self):
        from vime.ray.rollout import _resolve_vllm_config

        args = Namespace(vllm_config=None, prefill_num_servers=None, rollout_num_gpus=0)

        config = _resolve_vllm_config(args)

        assert len(config.models) == 1
        assert config.models[0].name == "default"
        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0

    def test_zero_gpu_config_takes_precedence_over_prefill_num_servers(self):
        from vime.ray.rollout import _resolve_vllm_config

        args = Namespace(vllm_config=None, prefill_num_servers=1, rollout_num_gpus=0)

        config = _resolve_vllm_config(args)

        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0

    def test_start_rollout_servers_zero_gpu_starts_router_without_engines(self, monkeypatch):
        from vime.ray import rollout as rollout_module

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456, None

        monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)
        args = Namespace(
            rollout_external=False,
            vllm_config=None,
            prefill_num_servers=None,
            rollout_num_gpus=0,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=False,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = rollout_module.start_rollout_servers(args, pg=(None, [], []))

        assert list(servers) == ["default"]
        assert init_handles == []
        server = servers["default"]
        assert server.router_ip == "127.0.0.1"
        assert server.router_port == 3456
        assert server.server_groups == []
        assert server.engines == []
        assert args.vllm_router_ip == "127.0.0.1"
        assert args.vllm_router_port == 3456
        assert args.vllm_model_routers == {"default": ("127.0.0.1", 3456)}

    def test_server_group_parallel_config_derives_tp_from_overridden_pp(self):
        from vime.ray.rollout import ServerGroup

        args = Namespace(
            num_gpus_per_node=8,
            vllm_pipeline_parallel_size=1,
            vllm_prefill_context_parallel_size=1,
            vllm_data_parallel_size=1,
            vllm_dp_size=1,
            vllm_enable_expert_parallel=True,
        )

        group = ServerGroup(
            args=args,
            pg=None,
            all_engines=[object()],
            num_gpus_per_engine=32,
            num_new_engines=1,
            vllm_overrides={"pipeline_parallel_size": 2},
        )

        assert group.parallel_config() == {
            "tp_size": 16,
            "pp_size": 2,
            "pcp_size": 1,
            "dp_size": 1,
            "enable_expert_parallel": True,
            "ep_size": 16,
        }

    def test_server_group_parallel_config_derives_tp_from_overridden_pcp(self):
        from vime.ray.rollout import ServerGroup

        args = Namespace(
            num_gpus_per_node=8,
            vllm_pipeline_parallel_size=1,
            vllm_prefill_context_parallel_size=1,
            vllm_data_parallel_size=1,
            vllm_dp_size=1,
            vllm_enable_expert_parallel=True,
        )
        group = ServerGroup(
            args=args,
            pg=None,
            all_engines=[object()],
            num_gpus_per_engine=8,
            num_new_engines=1,
            vllm_overrides={"prefill_context_parallel_size": 2, "data_parallel_size": 2},
        )

        assert group.parallel_config() == {
            "tp_size": 2,
            "pp_size": 1,
            "pcp_size": 2,
            "dp_size": 2,
            "enable_expert_parallel": True,
            "ep_size": 8,
        }

    def test_vllm_server_args_derive_tp_from_overridden_pp(self, monkeypatch):
        from vime.backends.vllm_utils import vllm_engine

        monkeypatch.setattr(vllm_engine, "_VLLM_SERVER_FIELDS", frozenset())

        args = Namespace(
            hf_checkpoint="/tmp/hf",
            seed=1,
            offload_rollout=False,
            rollout_num_gpus_per_engine=32,
            num_gpus_per_node=8,
            vllm_pipeline_parallel_size=1,
            vllm_prefill_context_parallel_size=1,
            vllm_data_parallel_size=1,
            vllm_dp_size=1,
            vllm_enable_expert_parallel=True,
            use_rollout_routing_replay=False,
            fp16=False,
            colocate=False,
            rollout_max_context_len=None,
            vllm_max_model_len=None,
        )

        kwargs, _ = vllm_engine._compute_server_args(
            args,
            rank=0,
            dist_init_addr="127.0.0.1:12345",
            host="127.0.0.1",
            port=30000,
            base_gpu_id=0,
            vllm_overrides={"pipeline_parallel_size": 2},
            num_gpus_per_engine=32,
        )

        assert kwargs["pipeline_parallel_size"] == 2
        assert kwargs["tensor_parallel_size"] == 16

    def test_offload_rollout_enables_vllm_sleep_mode(self, monkeypatch):
        from vime.backends.vllm_utils import vllm_engine

        monkeypatch.setattr(vllm_engine, "_VLLM_SERVER_FIELDS", frozenset())

        args = Namespace(
            hf_checkpoint="/tmp/hf",
            seed=1,
            offload_rollout=True,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            vllm_pipeline_parallel_size=1,
            vllm_prefill_context_parallel_size=1,
            vllm_data_parallel_size=1,
            vllm_dp_size=1,
            vllm_enable_expert_parallel=False,
            use_rollout_routing_replay=False,
            fp16=False,
            colocate=False,
            rollout_max_context_len=None,
            vllm_max_model_len=None,
            vllm_enable_sleep_mode=False,
        )
        compute_kwargs = {
            "rank": 0,
            "dist_init_addr": "127.0.0.1:12345",
            "host": "127.0.0.1",
            "port": 30000,
            "base_gpu_id": 0,
        }

        kwargs, _ = vllm_engine._compute_server_args(args, **compute_kwargs)
        assert kwargs["enable_sleep_mode"] is True
        assert args.vllm_enable_sleep_mode is True

    def test_start_rollout_servers_defers_engine_wait(self, monkeypatch):
        from vime.ray import rollout as rollout_module

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456, None

        def fake_start_engines(self, port_cursors=None):
            self.all_engines = [object() for _ in self.all_engines]
            return [f"init-{self.rank_offset}"], port_cursors or {}

        ray_get_calls = []

        def fake_ray_get(refs):
            ray_get_calls.append(refs)

        monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)
        monkeypatch.setattr(rollout_module.ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(rollout_module.ray, "get", fake_ray_get)

        args = Namespace(
            rollout_external=False,
            vllm_config=None,
            prefill_num_servers=None,
            rollout_num_gpus=2,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=False,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = rollout_module.start_rollout_servers(args, pg=(None, [], []))

        assert list(servers) == ["default"]
        assert init_handles == ["init-0"]
        assert ray_get_calls == []

    def test_start_rollout_servers_waits_for_epd_encoder_before_non_encoder(self, monkeypatch):
        from vime.backends.vllm_utils.vllm_config import ModelConfig, ServerGroupConfig, VllmConfig
        from vime.ray import rollout as rollout_module

        class FakeRemoteMethod:
            def __init__(self, value):
                self.value = value

            def remote(self):
                return self.value

        class FakeEngine:
            def __init__(self, url_ref):
                self.get_url = FakeRemoteMethod(url_ref)

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456, None

        def fake_resolve_vllm_config(args):
            return VllmConfig(
                models=[
                    ModelConfig(
                        name="default",
                        server_groups=[
                            ServerGroupConfig(worker_type="encoder", num_gpus=1),
                            ServerGroupConfig(worker_type="regular", num_gpus=1),
                        ],
                    )
                ]
            )

        def fake_start_engines(self, port_cursors=None):
            if self.worker_type == "encoder":
                self.all_engines = [FakeEngine("encoder-url-ref") for _ in self.all_engines]
            else:
                self.all_engines = [object() for _ in self.all_engines]
            return [f"{self.worker_type}-init-{self.rank_offset}"], port_cursors or {}

        ray_get_calls = []

        def fake_ray_get(refs):
            ray_get_calls.append(refs)
            if refs == ["encoder-url-ref"]:
                return ["http://encoder"]
            return None

        monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)
        monkeypatch.setattr(rollout_module, "_resolve_vllm_config", fake_resolve_vllm_config)
        monkeypatch.setattr(rollout_module.ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(rollout_module.ray, "get", fake_ray_get)

        args = Namespace(
            rollout_external=False,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=False,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = rollout_module.start_rollout_servers(args, pg=(None, [], []))

        groups = servers["default"].server_groups
        assert [group.worker_type for group in groups] == ["encoder", "regular"]
        assert groups[1].vllm_overrides["language_only"] is True
        assert groups[1].vllm_overrides["encoder_urls"] == ["http://encoder"]
        assert init_handles == ["regular-init-1"]
        assert ray_get_calls == [["encoder-init-0"], ["encoder-url-ref"]]


class TestGetModelUrl:
    def test_get_model_url_basic(self):
        """get_model_url should return the correct URL for a named model."""
        from argparse import Namespace

        from vime.rollout.vllm_rollout import get_model_url

        args = Namespace(
            vllm_router_ip="10.0.0.1",
            vllm_router_port=3000,
            vllm_model_routers={
                "actor": ("10.0.0.1", 3000),
                "ref": ("10.0.0.1", 3001),
            },
        )
        assert get_model_url(args, "actor") == "http://10.0.0.1:3000/inference/v1/generate"
        assert get_model_url(args, "ref") == "http://10.0.0.1:3001/inference/v1/generate"
        assert get_model_url(args, "ref", "/v1/chat/completions") == "http://10.0.0.1:3001/v1/chat/completions"

    def test_get_model_url_fallback(self):
        """get_model_url should fall back to default router if model not found."""
        from argparse import Namespace

        from vime.rollout.vllm_rollout import get_model_url

        args = Namespace(
            vllm_router_ip="10.0.0.1",
            vllm_router_port=3000,
            vllm_model_routers={"actor": ("10.0.0.1", 3000)},
        )
        assert get_model_url(args, "unknown") == "http://10.0.0.1:3000/inference/v1/generate"

    def test_get_model_url_no_routers(self):
        """get_model_url should work when model_routers is not set."""
        from argparse import Namespace

        from vime.rollout.vllm_rollout import get_model_url

        args = Namespace(
            vllm_router_ip="10.0.0.1",
            vllm_router_port=3000,
        )
        assert get_model_url(args, "anything") == "http://10.0.0.1:3000/inference/v1/generate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
