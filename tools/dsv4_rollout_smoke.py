"""Load a local DeepSeek-V4-Flash checkpoint and run a short vLLM rollout."""

import argparse

import numpy as np
from vllm import LLM, SamplingParams

from vime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--return-routed-experts", action="store_true")
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
        enable_return_routed_experts=args.return_routed_experts,
    )
    result = llm.generate(
        ["The capital of France is"],
        SamplingParams(max_tokens=4, temperature=0, logprobs=1, prompt_logprobs=1),
    )[0]
    print(f"tokens={result.outputs[0].token_ids} logprobs={result.outputs[0].logprobs}")
    if args.return_routed_experts:
        routes = result.outputs[0].routed_experts
        token_ids = list(result.prompt_token_ids) + list(result.outputs[0].token_ids[:-1])
        assert routes is not None and routes.shape == (len(token_ids), 43, 6), (
            None if routes is None else routes.shape
        )
        reader = SafetensorReader(args.checkpoint)
        for layer_id in range(3):
            expected = reader.get_tensor(f"layers.{layer_id}.ffn.gate.tid2eid")[token_ids].numpy()
            np.testing.assert_array_equal(routes[:, layer_id], expected)
        print(f"routed_experts={routes.shape} hash_layers=matched")


if __name__ == "__main__":
    main()
