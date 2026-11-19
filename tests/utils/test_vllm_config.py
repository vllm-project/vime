"""Unit tests for VllmConfig multi-model parsing and get_model_url."""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

NUM_GPUS = 0


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
@pytest.mark.parametrize("gpus_per_engine", [4, 16])
def test_pd_bootstrap_ports_follow_logical_engines(monkeypatch, worker_type, gpus_per_engine):
    from vime.backends.vllm_utils import engine_group

    actors = []
    for rank in range(4):
        actor = Mock()
        node_index = rank // 2 if gpus_per_engine == 4 else rank
        host = f"10.0.0.{node_index + 1}"
        actor._get_current_node_ip_and_free_port.remote.side_effect = (
            lambda start_port=15000, consecutive=1, host=host, node_index=node_index: (
                host,
                start_port + node_index * 100,
            )
        )
        actors.append((rank, actor))
    monkeypatch.setattr(engine_group.ray, "get", lambda value: value)
    addresses, _ = engine_group._allocate_rollout_engine_addr_and_ports_normal(
        args=Namespace(num_gpus_per_node=8, vllm_dp_size=1),
        rollout_engines=actors,
        worker_type=worker_type,
        num_gpus_per_engine=gpus_per_engine,
    )
    ports = [addresses[rank]["disaggregation_bootstrap_port"] for rank in range(4)]
    if gpus_per_engine == 4:
        assert ports[0] != ports[1]
        assert ports[2] != ports[3]
    else:
        assert ports[0] == ports[1]
        assert ports[2] == ports[3]
        assert addresses[0]["dist_init_addr"] == addresses[1]["dist_init_addr"]
        assert addresses[2]["dist_init_addr"] == addresses[3]["dist_init_addr"]


@pytest.mark.parametrize("node_rank", [0, 1])
def test_isolated_server_launch_waits_for_node_zero(monkeypatch, node_rank):
    from vime.backends.vllm_utils import vllm_engine

    executable = "/path with spaces/serve-python.sh"
    engine = vllm_engine.VLLMEngine(Namespace(vllm_python_executable=executable), rank=node_rank)
    engine.node_rank = node_rank
    engine.server_host = "[::1]"
    engine.server_port = 15000
    engine._register_to_router = Mock()
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    health = Mock()
    monkeypatch.setattr(vllm_engine, "launch_server_subprocess", launch)
    monkeypatch.setattr(vllm_engine, "_wait_server_healthy", health)
    monkeypatch.setattr(vllm_engine, "vllm_model_config_overrides", lambda path, overrides: overrides)
    monkeypatch.setattr(vllm_engine, "prepare_server_environment", lambda environ: None)
    monkeypatch.setattr(vllm_engine, "_build_subprocess_env", lambda args: {"CUDA_VISIBLE_DEVICES": "2"})
    monkeypatch.setattr(
        vllm_engine.multiprocessing, "Process", Mock(side_effect=AssertionError("wrong Python environment"))
    )
    args = {"model": "/checkpoint", "host": "[::1]", "port": 15000, "node_rank": node_rank, "_args": engine.args}
    engine._init_normal(args)
    assert engine.process is process
    launch.assert_called_once_with(
        executable,
        {key: value for key, value in {**args, "host": "::1"}.items() if not key.startswith("_")},
        environ={"CUDA_VISIBLE_DEVICES": "2"},
    )
    engine._register_to_router.assert_called_once_with(args)
    if node_rank == 0:
        assert health.call_args.kwargs["base_url"] == "http://[::1]:15000"
        assert health.call_args.kwargs["is_process_alive"]()
        process.poll.return_value = 1
        assert not health.call_args.kwargs["is_process_alive"]()
    else:
        health.assert_not_called()


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))

import _unit_stubs

_unit_stubs.install_vllm_cli_stubs()


def _write_yaml(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.flush()
    return f.name


class TestVllmConfigUpdateWeights:
    @pytest.mark.parametrize("first_model", ["regular", "pd", "epd"])
    @pytest.mark.parametrize("second_model", ["regular", "pd", "epd"])
    def test_models_starting_concurrently_use_disjoint_ports(self, monkeypatch, first_model, second_model):
        from vime.backends.vllm_utils import deployment, engine_group
        from vime.backends.vllm_utils.vllm_config import ModelConfig, ServerGroupConfig, VllmConfig

        layouts = {
            "regular": [("regular", 4)],
            "pd": [("prefill", 2), ("decode", 2)],
            "epd": [("encoder", 1), ("prefill", 1), ("decode", 2)],
        }
        config = VllmConfig(
            models=[
                ModelConfig(
                    name=name,
                    update_weights=(name == "actor"),
                    server_groups=[
                        ServerGroupConfig(worker_type=worker_type, num_gpus=num_gpus)
                        for worker_type, num_gpus in layouts[layout]
                    ],
                )
                for name, layout in (("actor", first_model), ("ref", second_model))
            ]
        )
        engines = []

        def create_engine(*args, **kwargs):
            engine = Mock()
            engine._get_current_node_ip_and_free_port.remote.side_effect = lambda start_port=10000, consecutive=1: (
                "127.0.0.1",
                start_port,
            )
            engine.get_url.remote.side_effect = (
                lambda: f"http://127.0.0.1:{engine.init.remote.call_args.kwargs['port']}"
            )
            engines.append(engine)
            return engine

        actor = Mock()
        actor.options.return_value.remote.side_effect = create_engine
        monkeypatch.setattr(engine_group.ray, "remote", lambda cls: actor)
        monkeypatch.setattr(engine_group.ray, "get", lambda refs: refs)
        monkeypatch.setattr(engine_group, "PlacementGroupSchedulingStrategy", lambda **kwargs: None)
        monkeypatch.setattr(deployment, "resolve_vllm_config", lambda _args: config)
        monkeypatch.setattr(
            deployment, "_start_router", lambda args, **kwargs: ("127.0.0.1", 3456 + int(kwargs["force_new"]), None)
        )
        args = Namespace(
            rollout_external=False,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            vllm_dp_size=1,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=True,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = deployment.start_rollout_servers(args, pg=(None, list(range(8)), list(range(8))))

        assert list(servers) == ["actor", "ref"]
        assert len(engines) == 8
        assert len(init_handles) == 8 - (first_model == "epd") - (second_model == "epd")
        reserved = set()
        for engine in engines:
            ports = engine.init.remote.call_args.kwargs
            dist_port = int(ports["dist_init_addr"].rsplit(":", 1)[1])
            requested = [ports["port"], ports["nccl_port"], *range(dist_port, dist_port + 30 + args.vllm_dp_size)]
            if "disaggregation_bootstrap_port" in ports:
                requested.append(ports["disaggregation_bootstrap_port"])
            assert len(requested) == len(set(requested))
            assert reserved.isdisjoint(requested), f"Reused ports: {reserved.intersection(requested)}"
            reserved.update(requested)

    @pytest.mark.parametrize("update_weights,level", [(True, 2), (False, 1)])
    @pytest.mark.parametrize("recover", [False, True])
    def test_offload_preserves_frozen_weights(self, monkeypatch, update_weights, level, recover):
        from vime.backends.vllm_utils import engine_group

        engine = Mock()
        group = engine_group.ServerGroup(
            args=Namespace(num_gpus_per_node=1),
            pg=None,
            all_engines=[None if recover else engine],
            num_gpus_per_engine=1,
            num_new_engines=0,
            needs_offload=True,
            model_path="frozen-model",
        )

        def start_engines(port_cursors):
            group.all_engines = [engine]
            group.num_new_engines = 1
            return [], port_cursors

        monkeypatch.setattr(group, "start_engines", start_engines)
        monkeypatch.setattr(engine_group.ray, "get", lambda handles: handles)
        server = engine_group.RolloutServer(server_groups=[group], update_weights=update_weights)
        if recover:
            server.recover()
        else:
            server.offload()
        engine.release_memory_occupation.remote.assert_called_once_with(level=level)

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
        config.models[0].resolve(Namespace(hf_checkpoint="/tmp/hf", rollout_num_gpus_per_engine=1))
        assert config.models[0].update_weights is True

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
        from vime.backends.vllm_utils.vllm_config import resolve_vllm_config

        args = Namespace(vllm_config=None, prefill_num_servers=None, rollout_num_gpus=0)

        config = resolve_vllm_config(args)

        assert len(config.models) == 1
        assert config.models[0].name == "default"
        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0

    def test_zero_gpu_config_takes_precedence_over_prefill_num_servers(self):
        from vime.backends.vllm_utils.vllm_config import resolve_vllm_config

        args = Namespace(vllm_config=None, prefill_num_servers=1, rollout_num_gpus=0)

        config = resolve_vllm_config(args)

        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0

    def test_start_rollout_servers_zero_gpu_starts_router_without_engines(self, monkeypatch):
        from vime.backends.vllm_utils import deployment

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456, None

        monkeypatch.setattr(deployment, "_start_router", fake_start_router)
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

        servers, init_handles = deployment.start_rollout_servers(args, pg=(None, [], []))

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
        from vime.backends.vllm_utils.engine_group import ServerGroup

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
        from vime.backends.vllm_utils.engine_group import ServerGroup

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
        from vime.utils import accelerator

        # Device mapping is tested separately; building server args needs no GPU.
        monkeypatch.setattr(accelerator, "resolve_visible_device_id", lambda device_id: device_id)
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
        from vime.utils import accelerator

        monkeypatch.setattr(accelerator, "resolve_visible_device_id", lambda device_id: device_id)
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
        from vime.backends.vllm_utils import deployment, disaggregation
        from vime.backends.vllm_utils.engine_group import ServerGroup

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456, None

        def fake_start_engines(self, port_cursors=None):
            self.all_engines = [object() for _ in self.all_engines]
            return [f"init-{self.rank_offset}"], port_cursors or {}

        def fail_if_waited(_refs):
            pytest.fail("regular deployment must not wait for engine initialization")

        monkeypatch.setattr(deployment, "_start_router", fake_start_router)
        monkeypatch.setattr(ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(disaggregation.ray, "get", fail_if_waited)

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

        servers, init_handles = deployment.start_rollout_servers(args, pg=(None, [], []))

        assert list(servers) == ["default"]
        assert init_handles == ["init-0"]

    def test_start_rollout_servers_routes_pd_to_disaggregated_deployment(self, monkeypatch):
        from vime.backends.vllm_utils import deployment
        from vime.backends.vllm_utils.engine_group import ServerGroup
        from vime.backends.vllm_utils.vllm_config import ModelConfig, ServerGroupConfig, VllmConfig

        def fake_start_router(
            args,
            *,
            has_pd_disaggregation=False,
            force_new=False,
        ):
            assert has_pd_disaggregation is True
            assert force_new is False
            return "127.0.0.1", 3456, None

        def fake_resolve_vllm_config(args):
            return VllmConfig(
                models=[
                    ModelConfig(
                        name="default",
                        server_groups=[
                            ServerGroupConfig(worker_type="prefill", num_gpus=1),
                            ServerGroupConfig(worker_type="decode", num_gpus=1),
                        ],
                    )
                ]
            )

        def fake_start_engines(self, port_cursors=None):
            assert (self.router_ip, self.router_port) == ("127.0.0.1", 3456)
            self.all_engines = [object() for _ in self.all_engines]
            return [f"{self.worker_type}-init-{self.rank_offset}"], port_cursors or {}

        monkeypatch.setattr(deployment, "_start_router", fake_start_router)
        monkeypatch.setattr(deployment, "resolve_vllm_config", fake_resolve_vllm_config)
        monkeypatch.setattr(ServerGroup, "start_engines", fake_start_engines)

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

        servers, init_handles = deployment.start_rollout_servers(args, pg=(None, [], []))

        groups = servers["default"].server_groups
        assert [group.worker_type for group in groups] == ["prefill", "decode"]
        assert init_handles == ["prefill-init-0", "decode-init-1"]
        assert all(group.router_ip == servers["default"].router_ip for group in groups)
        assert all(group.router_port == servers["default"].router_port for group in groups)

    def test_start_rollout_servers_waits_for_epd_encoder_before_non_encoder(self, monkeypatch):
        from vime.backends.vllm_utils import deployment, disaggregation
        from vime.backends.vllm_utils.engine_group import ServerGroup
        from vime.backends.vllm_utils.vllm_config import ModelConfig, ServerGroupConfig, VllmConfig

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

        monkeypatch.setattr(deployment, "_start_router", fake_start_router)
        monkeypatch.setattr(deployment, "resolve_vllm_config", fake_resolve_vllm_config)
        monkeypatch.setattr(ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(disaggregation.ray, "get", fake_ray_get)

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

        servers, init_handles = deployment.start_rollout_servers(args, pg=(None, [], []))

        groups = servers["default"].server_groups
        assert [group.worker_type for group in groups] == ["encoder", "regular"]
        assert "language_only" not in groups[1].vllm_overrides
        assert "mm_encoder_only" not in groups[1].vllm_overrides
        assert "encoder_urls" not in groups[1].vllm_overrides
        assert args.vllm_model_encoder_endpoints["default"][1] == ["http://encoder"]
        assert groups[1].vllm_overrides["ec_transfer_config"]["ec_connector"] == "ECCPUConnector"
        assert groups[1].vllm_overrides["ec_transfer_config"]["ec_connector_extra_config"] == {
            "ec_cpu_bytes": 1_000_000_000,
            "ec_enable_nixl": True,
        }
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
        """get_model_url should work when vllm_model_routers is not set."""
        from argparse import Namespace

        from vime.rollout.vllm_rollout import get_model_url

        args = Namespace(
            vllm_router_ip="10.0.0.1",
            vllm_router_port=3000,
        )
        assert get_model_url(args, "anything") == "http://10.0.0.1:3000/inference/v1/generate"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
