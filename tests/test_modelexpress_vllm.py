import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

if "cloudpickle" not in sys.modules:
    cloudpickle = types.ModuleType("cloudpickle")
    cloudpickle.dumps = lambda value: b"stub"
    sys.modules["cloudpickle"] = cloudpickle
_unit_stubs.install_vllm_cli_stubs()

from vime.backends.vllm_utils import vllm_engine
from vime.backends.vllm_utils.vllm_engine import VLLMEngine

pytestmark = pytest.mark.unit


def test_modelexpress_proxy_sends_exact_target_through_vllm_weight_transfer(monkeypatch):
    engine = VLLMEngine.__new__(VLLMEngine)
    engine.node_rank = 0
    calls = []
    monkeypatch.setattr(
        engine,
        "_make_request",
        lambda endpoint, payload=None: calls.append((endpoint, payload)),
    )

    engine.update_weights({"version_id": "a1b2c3d4"})

    assert calls == [("update_weights", {"update_info": {"version_id": "a1b2c3d4"}})]


def test_modelexpress_selects_vllm_backend(monkeypatch):
    args = SimpleNamespace(
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        colocate=False,
        debug_rollout_only=False,
        fp16=False,
        num_gpus_per_node=8,
        offload_rollout=False,
        rollout_num_gpus_per_engine=1,
        seed=1,
        update_weight_transport="modelexpress",
        use_critic=False,
        use_rollout_routing_replay=False,
        vllm_data_parallel_size=1,
        vllm_dp_size=1,
        vllm_pipeline_parallel_size=1,
    )
    vars(args)["hf_checkpoint"] = "/models/model"
    monkeypatch.setattr(vllm_engine, "_VLLM_SERVER_FIELDS", frozenset())

    server_args, _ = vllm_engine._compute_server_args(
        args,
        rank=0,
        dist_init_addr=None,
        host="127.0.0.1",
        port=30000,
    )

    assert server_args["weight_transfer_config"] == {"backend": "modelexpress"}
