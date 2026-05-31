import logging
import os
from typing import Any

from tau_bench.types import RunConfig

from vime.utils.types import Sample

logger = logging.getLogger(__name__)

TAU_CONFIGS = {
    "env": "retail",
    "agent": "tool-calling",
    "user_model": "openai/local-qwen3-4b",
    "user_model_provider": "openai",
    "task_split": "train",
    "user_strategy": "llm",
    "model_provider": "auto_router",
    "model": "qwen3-4b",
}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def res_to_sample(res: dict, task_index: int) -> Sample:
    status_mapping = {
        "completed": "completed",
        "truncated": "truncated",
        "aborted": "aborted",
    }
    status = status_mapping.get(res.get("status"), "aborted")

    sample = Sample(
        index=task_index,
        prompt=res.get("prompt", ""),
        tokens=res.get("tokens"),
        response=res.get("response", ""),
        reward=res.get("reward", 0.0),
        loss_mask=res.get("loss_mask"),
        status=status,
        metadata=res.get("info", {}),
    )

    if res.get("response_length") is not None:
        sample.response_length = res["response_length"]
    elif res.get("loss_mask"):
        sample.response_length = len(res["loss_mask"])
    elif res.get("tokens"):
        sample.response_length = len(res["tokens"])
    else:
        sample.response_length = 0

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    from .rollout import generate as rollout_generate

    return await rollout_generate(args, sample, sampling_params)
