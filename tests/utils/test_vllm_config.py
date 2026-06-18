"""Unit tests for VllmConfig multi-model parsing and get_model_url."""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

_unit_stubs.install_rollout_optional_stubs()


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
