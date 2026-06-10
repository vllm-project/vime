"""Pure ai21-evaluators integration, ported from ai21-verl (verl/ai21/reward/evaluators.py).

This module is framework-agnostic: it knows nothing about vime's ``Sample`` or verl's
``DataProto``. It owns the actual call into the ``ai21-evaluators`` library
(``run_verifiable_task_evaluations``) plus evaluator registration/warmup.

The vime reward function in ``ai21_evaluators.py`` builds the request from a ``Sample`` and
calls :func:`run_ai21_evaluation` here.

The ``ai21-evaluators`` / ``alignment-core`` libraries are pulled from AI21's private index
(see ``requirements_ai21.txt`` in ai21-verl). Their imports are kept at module top to match
upstream; importing this module therefore requires those deps installed.
"""

import asyncio
import gc
import os
import time
import traceback
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

# Must be set before ``ai21_evaluators`` is imported: they control whether CPU-bound evaluator
# work is dispatched to an internal process pool and how many workers it uses. Ported from
# ai21-verl's AI21RewardManager._AI21_EVALUATORS_ENV_VARS (which set these on the Ray actor).
os.environ.setdefault("AI21_EVALUATORS_SEND_CPU_BOUND_TO_PROCESS_POOL", "True")
os.environ.setdefault(
    "AI21_EVALUATORS_NUM_CPU_BOUND_WORKERS",
    str(min(os.cpu_count() or 1, 50)),
)

from ai21_evaluators.evaluators.registry import register_all_evaluators, warmup_registered_evaluators
from ai21_evaluators.evaluators.structures import DetailsType
from ai21_evaluators.file_formats.verifiable_task_dataset import VerifiableTask, run_verifiable_task_evaluations
from filelock import FileLock
from pydantic import BaseModel, Field

VERBOSE = os.environ.get("AI21_VERL_EVALUATORS_VERBOSE", "False").lower() == "true"


class AI21EvaluationRequest(BaseModel):
    task: VerifiableTask
    completion: str
    force_thinking: bool


class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    AI21_EVALUATORS_ERROR = "ai21_evaluators_error"
    VERL_ERROR = "verl_error"


class AI21EvaluationResponse(BaseModel):
    status: EvaluationStatus

    score: float | None = None
    details: DetailsType | None = None

    performance_metrics: dict[str, Any] = Field(default_factory=dict)


EXCLUDED_SCORE = 0.0
MISSING_THINK_END_TASK_SCORE = 0.0
MANUAL_GC = os.environ.get("AI21_VERL_EVALUATORS_MANUAL_GC", "False").lower() == "true"


async def run_ai21_evaluation(
    request: AI21EvaluationRequest,
    timeout: float,
) -> AI21EvaluationResponse:
    """Evaluate a single completion against a VerifiableTask's evaluators.

    Args:
        request: task + completion + force_thinking flag.
        timeout: Timeout in seconds for the (aggregated) evaluator call.

    Returns:
        AI21EvaluationResponse with status, score, details and performance metrics.
    """
    if request.force_thinking:
        # In force_thinking the opening <think> comes from the chat template; ensure the
        # closing </think> exists exactly once in the response.
        if request.completion.count("</think>") != 1:
            _log_if_verbose(
                f"encountered example with force_thinking but no </think> in completion. "
                f"settings score to {MISSING_THINK_END_TASK_SCORE}"
            )
            return AI21EvaluationResponse(
                status=EvaluationStatus.SUCCESS,
                score=MISSING_THINK_END_TASK_SCORE,
            )

    task_start_time = time.time()
    performance_metrics: dict[str, Any] = {
        "task_start_time": task_start_time,
        "evaluator_count": len(request.task.evaluations),
        "individual_evaluator_times": {},
    }

    response_details = None

    try:
        _log_if_verbose(
            {
                "category": "[EVALUATORS_TIME]",
                "description": f"Starting evaluation with timeout {timeout} seconds",
                "task_id": request.task.id,
                "completion": request.completion,
            }
        )

        start_time = time.time()

        # asyncio.timeout (not wait_for) is necessary to actually cancel the evaluation coroutine.
        async with asyncio.timeout(timeout):
            responses = await run_verifiable_task_evaluations(
                example=request.task,
                completion=request.completion,
            )
        assert len(responses) == 1, f"Expected single aggregated response, got {len(responses)} responses"
        response = responses[0]

        end_time = time.time()
        evaluation_time = end_time - start_time
        performance_metrics["evaluation_time"] = evaluation_time
        performance_metrics["avg_time_per_evaluator"] = (
            evaluation_time / len(request.task.evaluations) if len(request.task.evaluations) > 0 else 0
        )

        _log_if_verbose(f"Evaluation time: {evaluation_time} seconds")

        if response.result is None:
            status = EvaluationStatus.AI21_EVALUATORS_ERROR
            score = EXCLUDED_SCORE
            _log_if_verbose(f"Error in evaluation, error_code={response.error_code}, excluding example")
        else:
            status = EvaluationStatus.SUCCESS
            score = response.result
            _log_if_verbose(f"Aggregated score={score}")
            for evaluator_name, evaluator_result in zip(
                response.details.aggregated_evaluators,
                response.details.aggregated_evaluators_results,
                strict=False,
            ):
                if hasattr(evaluator_result, "processing_seconds"):
                    performance_metrics["individual_evaluator_times"][
                        evaluator_name
                    ] = evaluator_result.processing_seconds

        response_details = response.details

    except (asyncio.TimeoutError, TimeoutError):
        status = EvaluationStatus.TIMEOUT
        score = EXCLUDED_SCORE
        timeout_duration = time.time() - start_time
        _log(
            f"asyncio.TimeoutError occurred after {timeout_duration:.3f}s "
            f"Evaluators involved: {[evaluator.evaluator_name for evaluator in request.task.evaluations]}"
            f"(timeout was {timeout}s)"
        )
        performance_metrics["timeout_occurred"] = True
        performance_metrics["timeout_duration"] = timeout_duration
        _log_timeout_error_if_verbose(request=request, timeout_duration=timeout_duration)
    except Exception as e:
        status = EvaluationStatus.VERL_ERROR
        score = EXCLUDED_SCORE

        _log(f"Caught exception: type={type(e).__name__}, str={str(e)}, repr={repr(e)}")
        performance_metrics["exception_details"] = str(e)
        _log_evaluation_exception(request=request, exception=e)
    finally:
        # Force garbage collection to clean up any leaked memory.
        if MANUAL_GC:
            gc.collect()

    performance_metrics["non_success_count"] = int(status != EvaluationStatus.SUCCESS)
    performance_metrics["timeout_count"] = int(status == EvaluationStatus.TIMEOUT)

    task_end_time = time.time()
    performance_metrics["task_end_time"] = task_end_time
    performance_metrics["total_task_time"] = task_end_time - task_start_time

    _log_if_verbose(f"Total task time: {performance_metrics['total_task_time']:.3f} seconds")

    return AI21EvaluationResponse(
        status=status,
        score=score,
        performance_metrics=performance_metrics,
        details=response_details,
    )


def initialize_ai21_evaluators():
    """Register and warm up all evaluators. Safe to call multiple times across processes."""
    with _time_if_verbose("register_all_evaluators"):
        register_all_evaluators()

    with _time_if_verbose("warmup_evaluators_with_lock"):
        _warmup_evaluators_with_lock()

    if VERBOSE:
        os.environ["MODELS_INFERENCE_VLLM_CLIENT_VERBOSE"] = "True"
        os.environ["OPENAI_VLLM_CLIENT_VERBOSE"] = "True"

    assert "VLLM_MODEL_INTERNAL_URL" not in os.environ and "VLLM_MODEL_INTERNAL_NAME" not in os.environ, (
        '"VLLM_MODEL_INTERNAL_NAME" or "VLLM_MODEL_INTERNAL_URL" should not be set directly, '
        "use jlm_model_name variable with format model_name@model_address"
    )


_DEFAULT_WARMUP_DIR = Path.home() / ".ai21_evaluators_warmup"
_WARMUP_DIR = Path(os.environ.get("AI21_EVALUATORS_WARMUP_DIR", _DEFAULT_WARMUP_DIR))


def _warmup_evaluators_with_lock():
    """Warm up all registered evaluators exactly once per shared filesystem using file locking.

    Triggers ``_prerun_import()`` for all evaluators via ``warmup_registered_evaluators()``,
    which loads lazy dependencies (e.g. sentence_splitting downloads NLTK data at import).
    """
    _WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = _WARMUP_DIR / ".lock"
    warmup_marker = _WARMUP_DIR / ".done"
    with FileLock(lock_file):
        if warmup_marker.exists():
            _log_if_verbose("Already warmed up by another process")
        else:
            _log_if_verbose("First process on lock, warming up...")
            warmup_registered_evaluators()
            warmup_marker.touch()
            _log_if_verbose("Completed successfully")


LOG_PREFIX = "[AI21_EVAL_INTEGR]"


@contextmanager
def _time_if_verbose(name: str):
    """Time a block, printing only if VERBOSE is enabled."""
    if not VERBOSE:
        yield
        return
    start = time.time()
    yield
    _log(f"{name} took {time.time() - start:.6f}s")


def _log_timeout_error_if_verbose(request: AI21EvaluationRequest, timeout_duration: float) -> None:
    if VERBOSE:
        _log_dict(
            data={
                "type": "timeout",
                "task": request.task,
                "completion": request.completion,
                "timeout_duration": timeout_duration,
            },
        )


def _log_evaluation_exception(request: AI21EvaluationRequest, exception: Exception) -> None:
    if VERBOSE:
        _log_dict(
            data={
                "type": "exception",
                "task": request.task,
                "completion": request.completion,
                "exception": exception,
            },
        )
        _log(f"Full stacktrace:\n{traceback.format_exc()}")


def _log_dict(data: dict[str, Any]) -> None:
    for key, value in data.items():
        _log(f"{key}: {value}")


def _log_if_verbose(message: Any) -> None:
    if VERBOSE:
        _log(message)


def _log(message: Any) -> None:
    print(f"{LOG_PREFIX} {message}")
