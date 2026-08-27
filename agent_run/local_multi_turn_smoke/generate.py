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


def _trace_dir() -> Path:
    path = Path(os.environ["VIME_LOCAL_SANDBOX_TRACE_DIR"]) / _instance_id.get()
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _traced_git_diff(sb, workdir: str) -> str:
    diff = await _git_diff(sb, workdir)
    (_trace_dir() / "solution.patch").write_text(diff)
    _, trajectory, _ = await sb.exec(f"cat {workdir}/.harness/trajectory.jsonl", user="agent")
    (_trace_dir() / "trajectory.jsonl").write_text(trajectory)
    return diff


async def _traced_run_evaluation(md: dict, *, diff_text: str, timeout_sec: int):
    result = await _run_evaluation(md, diff_text=diff_text, timeout_sec=timeout_sec)
    (_trace_dir() / "grading.json").write_text(
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
        (_trace_dir() / "input.json").write_text(
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
