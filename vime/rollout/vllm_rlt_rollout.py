"""Native generation and reward evaluation through the ordinary Sample contract."""

import asyncio
from functools import lru_cache

import ray
from transformers import AutoTokenizer

from vime.rollout.base_types import RolloutFnTrainOutput
from vime.rollout.rm_hub import async_rm


@lru_cache(maxsize=1)
def _tokenizer(checkpoint: str):
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        raise ValueError("Native RLT evaluation requires an explicit evaluation recipe")
    groups = data_source.get_samples(args.rollout_batch_size)
    count = args.n_samples_per_prompt
    if len(groups) != args.rollout_batch_size or any(len(group) != count for group in groups):
        raise ValueError("Native rollout requires complete prompt groups")
    samples = [sample for group in groups for sample in group]
    identities = [(sample.group_index, sample.index) for sample in samples]
    tokenizer = _tokenizer(args.hf_checkpoint)
    for sample in samples:
        if sample.multimodal_inputs:
            raise ValueError("Native recurrent rollout supports text prompts")
        prompt = (
            tokenizer.apply_chat_template(
                sample.prompt, tokenize=False, add_generation_prompt=True, **sample.apply_chat_template_kwargs
            )
            if args.apply_chat_template
            else sample.prompt
        )
        sample.tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not sample.tokens:
            raise ValueError("Native rollout requires nonempty prompt tokens")
    generated = ray.get(args.rlt_engine.generate.remote(samples, rollout_id))
    if [(sample.group_index, sample.index) for sample in generated] != identities:
        raise ValueError("Native rollout must preserve every prompt group and sample index in order")
    for sample in generated:
        sample.response = tokenizer.decode(sample.tokens[-sample.response_length :], skip_special_tokens=True)

    async def reward():
        return await asyncio.gather(*(async_rm(args, sample) for sample in generated))

    for sample, value in zip(generated, asyncio.run(reward()), strict=True):
        sample.reward = value
    return RolloutFnTrainOutput(samples=[generated[i : i + count] for i in range(0, len(generated), count)])
