import json
import os
from contextvars import ContextVar
from pathlib import Path

from examples.coding_agent_rl import generate as coding_generate
from examples.coding_agent_rl import swe

from .sandbox import LocalDockerSandbox

coding_generate.E2BSandbox = LocalDockerSandbox
swe.E2BSandbox = LocalDockerSandbox

_git_diff = swe.git_diff
_run_evaluation = swe.run_evaluation
_instance_id: ContextVar[str] = ContextVar("instance_id", default="unknown")


def _trace_dir() -> Path | None:
    """Per-instance trace directory, or None when tracing is off.

    sandbox.py already treats VIME_LOCAL_SANDBOX_TRACE_DIR as optional and skips
    tracing when it is unset; this module required it and raised KeyError, so
    running without it killed the rollout over a debugging aid.
    """
    root = os.environ.get("VIME_LOCAL_SANDBOX_TRACE_DIR")
    if not root:
        return None
    path = Path(root) / _instance_id.get()
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _traced_git_diff(sb, workdir: str) -> str:
    diff = await _git_diff(sb, workdir)
    td = _trace_dir()
    if td is not None:
        (td / "solution.patch").write_text(diff)
        _, trajectory, _ = await sb.exec(f"cat {workdir}/.harness/trajectory.jsonl", user="agent")
        (td / "trajectory.jsonl").write_text(trajectory)
    return diff


async def _traced_run_evaluation(md: dict, *, diff_text: str, timeout_sec: int):
    result = await _run_evaluation(md, diff_text=diff_text, timeout_sec=timeout_sec)
    td = _trace_dir()
    if td is not None:
        (td / "grading.json").write_text(
            json.dumps(
                {
                    "instance_id": md["instance_id"],
                    "reward": result.reward,
                    "applied_cleanly": result.applied_cleanly,
                    "eval_cmd": md["grading"].get("eval_cmd"),
                },
                indent=2,
            )
            + "\n"
        )
    return result


swe.git_diff = _traced_git_diff
swe.run_evaluation = _traced_run_evaluation


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    token = _instance_id.set(base_sample.metadata["instance_id"])
    try:
        td = _trace_dir()
        if td is not None:
            (td / "input.json").write_text(
                json.dumps(
                    {
                        "prompt": base_sample.prompt,
                        "label": base_sample.label,
                        "metadata": base_sample.metadata,
                        "sampling_params": sampling_params,
                        "evaluation": evaluation,
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
        return await coding_generate.generate(args, base_sample, sampling_params, evaluation)
    finally:
        _instance_id.reset(token)
