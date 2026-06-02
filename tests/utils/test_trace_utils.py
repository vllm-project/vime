import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vime.utils.trace_utils import trace_span
from vime.utils.types import Sample


def _load_trace_timeline_viewer_module():
    module_path = Path(__file__).resolve().parents[2] / "tools" / "trace_timeline_viewer.py"
    module_name = "test_trace_timeline_viewer_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_trace_timeline_viewer_omits_virtual_pd_lanes_without_pd_attrs(tmp_path: Path):
    viewer = _load_trace_timeline_viewer_module()
    sample = Sample(index=0, prompt="hello")

    with trace_span(sample, "vllm_generate", attrs={"max_new_tokens": 8}) as span:
        span.update(
            {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "cached_tokens": 1,
                "finish_reason": "stop",
            }
        )

    pt_path = tmp_path / "rollout.pt"
    torch.save({"samples": [sample]}, pt_path)

    cache = viewer._build_cache_data(pt_path)

    assert cache["sample_count"] == 1
    row = cache["rows"][0]
    assert row["lane_count"] == 1
    assert row["item_count"] == 1
    assert row["closed_span_count"] == 1

    item = row["items"][0]
    assert item["name"] == "vllm_generate"
    assert item["attrs"]["end_attrs"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "cached_tokens": 1,
        "finish_reason": "stop",
    }
    assert "[P]" not in item["name"]
    assert "[D]" not in item["name"]
