"""Load a local DeepSeek-V4-Flash checkpoint and run a short vLLM rollout."""

import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()

    llm = LLM(
        model=args.checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        moe_backend="triton",
        trust_remote_code=True,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="fp8",
        block_size=256,
        enforce_eager=True,
    )
    result = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=4, temperature=0, logprobs=1, prompt_logprobs=1),
    )[0]
    print(f"tokens={result.outputs[0].token_ids} logprobs={result.outputs[0].logprobs}")


if __name__ == "__main__":
    main()
