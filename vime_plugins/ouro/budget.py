"""Deterministic global-depth plans and strict on-policy group identities."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BudgetSchedule:
    depths: tuple[int, ...] = (2, 3, 4)

    def __post_init__(self) -> None:
        if not self.depths or any(type(k) is not int or k not in (2, 3, 4) for k in self.depths):
            raise ValueError("Budget schedule must contain supported Ouro depths 2/3/4")

    def at(self, update: int) -> int:
        if update < 0:
            raise ValueError("Update must be nonnegative")
        return self.depths[update % len(self.depths)]


@dataclass(frozen=True)
class ExecutionPlan:
    loop_budget: int
    policy_version: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.loop_budget not in (2, 3, 4) or type(self.loop_budget) is not int:
            raise ValueError("Supported Ouro budgets are 2/3/4")
        if self.policy_version < 0 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid version or sampling temperature")

    @property
    def config_hash(self) -> str:
        config = {
            "prefill": self.loop_budget,
            "decode": self.loop_budget,
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "early_exit": False,
            "logprob_space": "temperature_scaled_full_vocab",
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def group_key(self, prompt_id: str) -> tuple[str, int, int, str]:
        return prompt_id, self.loop_budget, self.policy_version, self.config_hash

    def metadata(self, prompt_id: str) -> dict:
        return {
            **asdict(self),
            "prompt_id": prompt_id,
            "prefill_loop_budget": self.loop_budget,
            "decode_loop_budget": self.loop_budget,
            "execution_config_hash": self.config_hash,
        }

    def validate(self, metadata: list[dict], group_size: int) -> None:
        if not metadata or group_size < 2 or len(metadata) % group_size:
            raise ValueError("GRPO requires complete groups with at least two completions")
        for offset in range(0, len(metadata), group_size):
            group = metadata[offset : offset + group_size]
            expected = self.metadata(group[0]["prompt_id"])
            if any(item != expected for item in group):
                raise ValueError("Mixed prompt/depth/version/execution policy in GRPO group")


def synchronize_plan(plan: ExecutionPlan) -> None:
    """Every DP rank must agree before executing a depth-dependent forward."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
    fingerprint = int(plan.config_hash[:15], 16)
    values = torch.tensor([plan.loop_budget, plan.policy_version, fingerprint], device=device)
    minimum, maximum = values.clone(), values.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise ValueError("DP ranks disagree on the recurrent execution plan")
