"""vime custom reward function backed by the ``ai21-evaluators`` library.

Wire it in with::

    --custom-rm-path vime_plugins.rm.ai21_evaluators.ai21_reward

It supports both vime reward paths (see ``vime/rollout/rm_hub/__init__.py``):
- single-sample (``async_rm``): called as ``ai21_reward(args, sample)``
- group/batched (``batched_async_rm``): called as ``ai21_reward(args, [sample, ...])``

The per-example evaluator spec is carried on ``sample.metadata`` (the AmalgamDataset / offline
data-prep step populates these from the ai21-evaluators VerifiableTask format):

    metadata["reward_model"]        list of evaluator entries (evaluator_name, evaluator_config,
                                    query_args, evaluation_id)
    metadata["aggregation_config"]  dict passed to AggregationConfig(**...)
    metadata["id"]                  task id (falls back to sample.index)
    metadata["messages"]            chat messages (falls back to sample.prompt)
    metadata["extra_info"]          arbitrary task metadata (optional)
    metadata["force_thinking"]      bool (optional, default False)
    metadata["jlm_model_name"]      "model_name@model_address" (optional)

Global config is read from ``args`` if present, else the matching env var, else a default
(so it works whether or not these args are added to vime's parser):

    ai21_evaluators_timeout       AI21_EVALUATORS_TIMEOUT        60.0
    ai21_evaluators_config_file   AI21_EVALUATORS_CONFIG_FILE    None
    ai21_clean_thinking_trace     AI21_CLEAN_THINKING_TRACE      False

Return value: a float by default. If ``--reward-key`` is set, returns a dict containing that key
plus extra fields (score, status, do_exclude) so the rest can be logged.
"""

import asyncio
import os
import threading
from typing import Any

from vime.utils.types import Sample

_init_lock = threading.Lock()
_initialized = False


def _ensure_initialized() -> None:
    """Register + warm up evaluators exactly once per process (thread-safe)."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        from .ai21_evaluators_lib import initialize_ai21_evaluators

        initialize_ai21_evaluators()
        _initialized = True


def _cfg(args, attr: str, env: str, default: Any) -> Any:
    """Resolve config from args, then env var, then default."""
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


def _completion_from_sample(args, sample: Sample) -> str:
    completion = sample.response or ""
    clean = str(_cfg(args, "ai21_clean_thinking_trace", "AI21_CLEAN_THINKING_TRACE", "False"))
    if clean.lower() == "true":
        completion = completion.split("</think>")[-1]
    return completion


def _build_request(args, sample: Sample):
    """Convert a vime ``Sample`` into an ``AI21EvaluationRequest``.

    Mirrors AI21RewardManager._data_proto_item_to_ai21_eval_request, sourcing fields from
    ``sample.metadata`` instead of a verl DataProto.
    """
    # Validate data shape before the heavy (private-dep) imports so bad data fails fast.
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    eval_entries = metadata.get("reward_model")
    if eval_entries is None:
        raise ValueError(
            "ai21_reward: sample.metadata is missing 'reward_model' (the list of evaluator entries). "
            "Ensure the dataset carries the ai21-evaluators VerifiableTask spec in metadata."
        )
    _validate_eval_entries(eval_entries)

    from ai21_evaluators.file_formats.verifiable_task_dataset import AggregationConfig, EvaluationEntry, VerifiableTask

    from .ai21_evaluators_lib import AI21EvaluationRequest

    task_id = metadata.get("id", sample.index)
    messages = metadata.get("messages", sample.prompt)
    aggregation_config = metadata.get("aggregation_config", {})
    task_metadata = metadata.get("extra_info") or {}
    force_thinking = bool(metadata.get("force_thinking", False))
    jlm_model_name = metadata.get("jlm_model_name")

    task = VerifiableTask(
        id=task_id,
        messages=messages,
        evaluations=[
            EvaluationEntry(
                evaluator_name=entry.get("evaluator_name"),
                evaluator_config=entry.get("evaluator_config", {}),
                query_args=entry.get("query_args", {}),
                evaluation_id=entry.get("evaluation_id"),
            )
            for entry in eval_entries
        ],
        metadata=task_metadata,
        aggregation_config=AggregationConfig(**aggregation_config),
    )

    _patch_task_in_place(
        task=task,
        ai21_evaluators_config_file=_cfg(args, "ai21_evaluators_config_file", "AI21_EVALUATORS_CONFIG_FILE", None),
        jlm_model_name=jlm_model_name,
    )

    return AI21EvaluationRequest(
        task=task,
        completion=_completion_from_sample(args, sample),
        force_thinking=force_thinking,
    )


async def _run_evaluation(request, timeout: float):
    """Run a single evaluation (split out so tests can monkeypatch it)."""
    from .ai21_evaluators_lib import run_ai21_evaluation

    return await run_ai21_evaluation(request=request, timeout=timeout)


def _format_reward(args, response) -> float | dict[str, Any]:
    """Shape the reward for vime's ``Sample.get_reward_value`` (float, or dict if --reward-key).

    ``EvaluationStatus`` is a ``str`` Enum whose SUCCESS value is "success"; comparing on the
    string value avoids importing the (private-dep) lib here so this stays unit-testable.
    """
    score = float(response.score if response.score is not None else 0.0)
    status_value = response.status.value if hasattr(response.status, "value") else response.status
    reward_key = getattr(args, "reward_key", None)
    if not reward_key:
        return score
    return {
        reward_key: score,
        "score": score,
        "status": status_value,
        "do_exclude": status_value != "success",
    }


async def _score_one(args, sample: Sample, **kwargs) -> float | dict[str, Any]:
    _ensure_initialized()
    timeout = float(_cfg(args, "ai21_evaluators_timeout", "AI21_EVALUATORS_TIMEOUT", 60.0))
    request = _build_request(args, sample)
    response = await _run_evaluation(request, timeout)
    return _format_reward(args, response)


async def ai21_reward(args, sample_or_samples, **kwargs):
    """Entry point for ``--custom-rm-path``. Handles both single Sample and list[Sample]."""
    if isinstance(sample_or_samples, list):
        return await asyncio.gather(*[_score_one(args, s, **kwargs) for s in sample_or_samples])
    return await _score_one(args, sample_or_samples, **kwargs)


def _validate_eval_entries(eval_entries) -> list:
    """Validate the evaluator entries on a sample (ported from AI21RewardManager)."""
    try:
        import numpy as np

        if isinstance(eval_entries, np.ndarray):
            eval_entries = eval_entries.tolist()
    except ImportError:
        pass

    assert len(eval_entries) > 0, "AI21 data error: example has no evaluators configured"

    eval_names = []
    for eval_entry in eval_entries:
        assert isinstance(eval_entry, dict), f"AI21 data error: evaluator entry must be dict, got {type(eval_entry)}"
        name = eval_entry.get("evaluation_id") or eval_entry.get("evaluator_name")
        assert name is not None, f"AI21 data error: evaluator without a name and id: {eval_entry}"
        eval_names.append(name)

    assert len(eval_names) == len(set(eval_names)), f"AI21 data error: duplicate evaluator names: {eval_names}"
    return eval_names


def _patch_task_in_place(task, ai21_evaluators_config_file, jlm_model_name) -> None:
    """Apply an evaluators config file and/or override the judge model name, in place."""
    from ai21_evaluators.file_formats.verifiable_task_dataset import (
        VerifiableTaskDataset,
        apply_evaluators_config_to_verifiable_dataset,
        change_verifiable_task_model_name,
    )

    one_task_dataset = VerifiableTaskDataset([task])

    if ai21_evaluators_config_file is not None:
        apply_evaluators_config_to_verifiable_dataset(
            config_file=ai21_evaluators_config_file,
            verifiable_task_dataset=one_task_dataset,
        )

    if jlm_model_name is not None:
        change_verifiable_task_model_name(
            verifiable_task=task,
            model_name=jlm_model_name,
        )
