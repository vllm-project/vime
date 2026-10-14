"""Run the experimental same-depth-KV, fixed-interval Ouro inference path."""

import argparse
import json
from dataclasses import asdict

import torch
from transformers import AutoTokenizer
from vllm_rlt.models.ouro import OuroForCausalLM

from vime_plugins.ouro.deferred_decode import generate_deferred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local Ouro checkpoint directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--responses", type=int, default=4)
    parser.add_argument("--interval", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--eager", action="store_true", help="Execute the same readout policy with immediate deep writes"
    )
    args = parser.parse_args()
    if args.responses <= 0:
        parser.error("--responses must be positive")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = OuroForCausalLM.from_pretrained(
        args.model, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).eval()
    prompt = tokenizer.encode(
        args.prompt + "\nGive your final answer in \\boxed{}.\n###Response\n", add_special_tokens=False
    )
    outputs = generate_deferred(
        model,
        [prompt] * args.responses,
        [args.seed + index for index in range(args.responses)],
        interval=args.interval,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        backend="triton" if device.type == "cuda" else "torch",
        defer=not args.eager,
    )
    print(
        json.dumps(
            [
                {**asdict(output), "text": tokenizer.decode(output.token_ids, skip_special_tokens=False)}
                for output in outputs
            ],
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
