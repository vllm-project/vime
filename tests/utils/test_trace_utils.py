import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vime.utils.trace_utils import TRACE_CHILDREN_KEY, build_vllm_meta_trace_attrs, trace_span
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
def test_build_vllm_meta_trace_attrs_keeps_standard_and_pd_fields():
    attrs = build_vllm_meta_trace_attrs(
        {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "cached_tokens": 3,
            "pd_prefill_forward_duration": 0.125,
            "pd_decode_transfer_duration": 0.05,
            "finish_reason": {"type": "stop"},
            "unused_field": "ignored",
        }
    )
    trace_children = attrs.pop(TRACE_CHILDREN_KEY)

    assert attrs == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "finish_reason": "stop",
    }
    assert trace_children[0]["name"] == "vllm_pd_prefill"
    assert trace_children[0]["children"][0]["attrs"] == {"pd_prefill_forward_duration": 0.125}
    assert trace_children[1]["name"] == "vllm_pd_decode"
    assert trace_children[1]["children"][0]["attrs"] == {"pd_decode_transfer_duration": 0.05}


@pytest.mark.unit
def test_build_vllm_meta_trace_attrs_reads_native_response_shape():
    assert build_vllm_meta_trace_attrs(
        {
            "choices": [{"finish_reason": "length"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "cached_tokens": 3},
        }
    ) == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "finish_reason": "length",
    }


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
