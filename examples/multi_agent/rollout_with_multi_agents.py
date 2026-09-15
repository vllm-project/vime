import random

from transformers import AutoTokenizer

from vime.utils.misc import load_function
from vime.utils.types import Sample

MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path": "examples.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,
    "incorrect_reward_weight": 0.8,
    "correct_reward_weight": 1.2,
}


async def generate_with_multi_agents(args, sample: Sample, sampling_params, evaluation=False) -> list[Sample]:

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    max_context_length = args.rollout_max_context_len if not evaluation else args.eval_max_context_len

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    for key, value in MULTI_AGENT_CONFIGS.items():
        setattr(args, key, value)

    custom_multi_agent_func = load_function(args.custom_multi_agent_function_path)
    samples = await custom_multi_agent_func(args, sample)

    # VIME compact rollouts return multiple training samples from one source
    # sample. Newer VIME requires all siblings to share a rollout_id so loss
    # reduction counts the source rollout once instead of over-counting agents.
    compact_rollout_id = (
        sample.rollout_id
        if sample.rollout_id is not None
        else (
            sample.index
            if sample.index is not None
            else sample.group_index if sample.group_index is not None else id(sample)
        )
    )
    for sibling in samples:
        sibling.rollout_id = compact_rollout_id

    random.shuffle(samples)

    return samples
