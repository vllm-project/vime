"""Exercise the real argument provider after merging disk and sparse sync."""

import argparse

import pytest

from vime.utils.arguments import get_vime_extra_args_provider


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["nccl", "disk", "sparse_hccl"])
def test_transport_options_are_registered_once(transport):
    parser = get_vime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(
        ["--rollout-batch-size", "1", "--update-weight-transport", transport,
         "--update-weight-delta-verify-every", "1"]
    )
    assert args.update_weight_transport == transport
    assert args.update_weight_delta_verify_every == 1
    assert args.update_weight_delta_batch_diff == 32
    assert args.update_weight_delta_batch_gather == 32
    assert args.update_weight_stage_timing is False


@pytest.mark.unit
def test_stage_timing_requires_explicit_flag():
    parser = get_vime_extra_args_provider()(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", "--update-weight-stage-timing"])
    assert args.update_weight_stage_timing is True
