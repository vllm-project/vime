"""Rank coverage and digest validation for the vLLM checker protocol."""

import pytest

from vime.backends.vllm_utils.weight_checker import expected_rank_prefixes, validate_checksums

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, {"dp0:pp0:pcp0:tp0"}),
        ({"tensor_parallel_size": 2}, {"dp0:pp0:pcp0:tp0", "dp0:pp0:pcp0:tp1"}),
        ({"data_parallel_size": 2}, {"dp0:pp0:pcp0:tp0", "dp1:pp0:pcp0:tp0"}),
        (
            {"data_parallel_size": 8, "data_parallel_index": 3, "data_parallel_external_lb": True},
            {"dp3:pp0:pcp0:tp0"},
        ),
        (
            {
                "data_parallel_size": 8,
                "data_parallel_index": 3,
                "data_parallel_hybrid_lb": True,
                "data_parallel_size_local": 2,
            },
            {"dp3:pp0:pcp0:tp0", "dp4:pp0:pcp0:tp0"},
        ),
        (
            {"data_parallel_size": 8, "data_parallel_index": 5, "data_parallel_rank_local": 0},
            {"dp5:pp0:pcp0:tp0"},
        ),
    ],
)
def test_expected_managed_rank_prefixes(config, expected):
    assert expected_rank_prefixes(config) == expected


def test_pipeline_and_context_ranks_are_included():
    ranks = expected_rank_prefixes({"pipeline_parallel_size": 2, "prefill_context_parallel_size": 2})
    assert ranks == {"dp0:pp0:pcp0:tp0", "dp0:pp0:pcp1:tp0", "dp0:pp1:pcp0:tp0", "dp0:pp1:pcp1:tp0"}


@pytest.mark.parametrize("config", [{"tensor_parallel_size": 0}, {"data_parallel_index": -1}])
def test_invalid_layout_is_rejected(config):
    with pytest.raises(ValueError, match="parallel configuration"):
        expected_rank_prefixes(config)


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"checksums": {}},
        {"checksums": {"weight": "a" * 64}},
        {"checksums": {"dp0:pp0:pcp0:tp0:w": "bad"}},
    ],
)
def test_invalid_checksums_are_rejected(response):
    with pytest.raises(ValueError):
        validate_checksums(response, ["dp0:pp0:pcp0:tp0"])


def test_missing_rank_is_rejected_before_reset():
    with pytest.raises(ValueError, match="missing=.*dp1"):
        validate_checksums({"checksums": {"dp0:pp0:pcp0:tp0:w": "a" * 64}}, ["dp0:pp0:pcp0:tp0", "dp1:pp0:pcp0:tp0"])


def test_validated_baseline_is_an_independent_dictionary():
    values = {"dp0:pp0:pcp0:tp0:w": "a" * 64}
    baseline = validate_checksums({"checksums": values}, ["dp0:pp0:pcp0:tp0"])
    values.clear()
    assert baseline == {"dp0:pp0:pcp0:tp0:w": "a" * 64}
