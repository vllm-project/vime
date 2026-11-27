# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker identity and all-engine checker responses without device execution."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from fastapi import HTTPException

from vllm.entrypoints.rl.online import api_router
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker import worker_base

PREFIX = "dp4:pp1:pcp1:tp1"


@pytest.fixture
def worker_factory(monkeypatch):
    for name in ("get_pp_group", "get_pcp_group", "get_tp_group"):
        monkeypatch.setattr(worker_base, name, lambda: SimpleNamespace(rank_in_group=1))

    def create(runner_api="get_model"):
        model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        model.register_buffer("cache", torch.tensor(7.0))
        worker = object.__new__(worker_base.WorkerBase)
        worker.parallel_config = SimpleNamespace(data_parallel_index=4, data_parallel_rank=0)
        worker.model_runner = (
            SimpleNamespace(get_model=lambda: model) if runner_api == "get_model" else SimpleNamespace(model=model)
        )
        return worker, model

    return create


@pytest.mark.parametrize("runner_api", ["get_model", "model"])
def test_shared_worker_checker_resets_and_restores_parameters(worker_factory, runner_api):
    worker, model = worker_factory(runner_api)
    original = model.weight.detach().clone()
    baseline = worker.compute_weight_checksums()
    assert set(baseline) == {f"{PREFIX}:weight"}
    worker.reset_weights()
    assert not model.weight.count_nonzero()
    assert model.cache.item() == 7
    assert worker.compute_weight_checksums() != baseline
    with torch.no_grad():
        model.weight.copy_(original)
    assert worker.compute_weight_checksums() == baseline


def test_worker_identity_is_stable_but_changes_on_replacement(worker_factory):
    worker, model = worker_factory()
    first = worker.get_weight_checker_state()
    assert first == worker.get_weight_checker_state()
    assert first["rank_prefix"] == PREFIX
    assert len(first["instance_id"]) == 32
    replacement, other_model = worker_factory()
    with torch.no_grad():
        other_model.weight.copy_(model.weight)
    second = replacement.get_weight_checker_state()
    assert first["checksums"] == second["checksums"]
    assert first["instance_id"] != second["instance_id"]


def test_checker_states_cover_all_dp_engines():
    states = [[{"worker": "dp0:tp0"}, {"worker": "dp0:tp1"}], [{"worker": "dp1:tp0"}]]
    core = SimpleNamespace(call_utility_all_async=AsyncMock(return_value=states))
    result = asyncio.run(AsyncLLM.get_weight_checker_states(SimpleNamespace(engine_core=core)))
    assert result == states[0] + states[1]
    core.call_utility_all_async.assert_awaited_once_with("collective_rpc", "get_weight_checker_state", None, (), None)


@pytest.mark.parametrize("action", ["checksum", "compare"])
@pytest.mark.parametrize("include_ids", [False, True])
def test_http_checker_keeps_legacy_protocol_and_opt_in_identity(monkeypatch, action, include_ids):
    checksums = {f"{PREFIX}:weight": "a" * 64}
    state = {"rank_prefix": PREFIX, "instance_id": "1" * 32, "checksums": checksums}
    client = SimpleNamespace(
        get_weight_checker_states=AsyncMock(return_value=[state]),
        compute_weight_checksums=AsyncMock(return_value=[checksums]),
    )
    monkeypatch.setattr(api_router, "engine_client", lambda request: client)
    response = asyncio.run(api_router.weight_checker(None, action, checksums, include_ids))
    body = json.loads(response.body)
    assert body == (
        ({"checksums": checksums} if action == "checksum" else {"match": True, "mismatches": []})
        | ({"worker_ids": {PREFIX: "1" * 32}} if include_ids else {})
    )
    if include_ids:
        client.get_weight_checker_states.assert_awaited_once_with()
        client.compute_weight_checksums.assert_not_awaited()
    else:
        client.compute_weight_checksums.assert_awaited_once_with()
        client.get_weight_checker_states.assert_not_awaited()


def test_http_checker_rejects_duplicate_worker_ranks(monkeypatch):
    state = {"rank_prefix": PREFIX, "instance_id": "1" * 32, "checksums": {}}
    client = SimpleNamespace(get_weight_checker_states=AsyncMock(return_value=[state, state]))
    monkeypatch.setattr(api_router, "engine_client", lambda request: client)
    with pytest.raises(HTTPException, match="Duplicate weight checker rank"):
        asyncio.run(api_router.weight_checker(None, "checksum", None, True))


def test_http_reset(monkeypatch):
    client = SimpleNamespace(collective_rpc=AsyncMock())
    monkeypatch.setattr(api_router, "engine_client", lambda request: client)
    response = asyncio.run(api_router.weight_checker(None, "reset", None, False))
    client.collective_rpc.assert_awaited_once_with("reset_weights")
    assert json.loads(response.body) == {"status": "reset"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
