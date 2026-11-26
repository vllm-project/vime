"""Single-node NPU integration test. Run from the repository root with
VIME_PLATFORM=npu python tests/test_qwen3_moe_ep2_weight_checker_npu.py
on a dedicated three-NPU allocation with the paired Megatron/vLLM runtimes.
"""

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def prepare(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

    model_dir = directory / "qwen3-moe"
    vocab = {"<unk>": 0, "<eos>": 1, **{str(i): i + 2 for i in range(126)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        rope_theta=10000,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(1234)
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    prompts = directory / "prompts.jsonl"
    prompts.write_text('{"prompt":"1 2","label":"3"}\n{"prompt":"2 3","label":"5"}\n')
    return model_dir, prompts


def main():
    import ray

    with tempfile.TemporaryDirectory(prefix="vime-ep2-") as tmp:
        model, prompts = prepare(Path(tmp))
        # Two training ranks own four experts each. A separate rollout worker
        # must receive both shards to restore its pre-reset parameter digests.
        args = f"""
            --train-backend megatron --bf16 --actor-num-nodes 1 --actor-num-gpus-per-node 2
            --num-gpus-per-node 3 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1
            --expert-model-parallel-size 2 --expert-tensor-parallel-size 1
            --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
            --num-layers 2 --hidden-size 256 --ffn-hidden-size 512 --num-attention-heads 4
            --group-query-attention --num-query-groups 2 --kv-channels 64 --qk-layernorm
            --swiglu --disable-bias-linear --normalization RMSNorm --norm-epsilon 1e-6
            --position-embedding-type rope --rotary-base 10000 --vocab-size 128
            --num-experts 8 --moe-layer-freq 1 --moe-ffn-hidden-size 256 --moe-router-topk 2
            --moe-grouped-gemm --moe-token-dispatcher-type alltoall --moe-router-dtype fp32
            --moe-router-score-function softmax --moe-aux-loss-coeff 0
            --hf-checkpoint {shlex.quote(str(model))} --load {shlex.quote(str(model))} --no-load-optim --no-load-rng
            --prompt-data {shlex.quote(str(prompts))} --input-key prompt --label-key label --rm-type math
            --num-rollout 1 --rollout-batch-size 2 --n-samples-per-prompt 2 --rollout-max-response-len 16
            --global-batch-size 4 --micro-batch-size 1 --seq-length 128 --use-dynamic-batch-size --max-tokens-per-gpu 512
            --advantage-estimator grpo --optimizer adam --lr 1e-6 --lr-decay-style constant
            --attention-dropout 0 --hidden-dropout 0 --attention-backend flash --no-gradient-accumulation-fusion
            --vllm-python-executable {shlex.quote(sys.executable)} --vllm-enforce-eager --vllm-max-model-len 128
            --vllm-max-num-seqs 4 --vllm-max-num-batched-tokens 128 --vllm-kv-cache-memory 268435456
            --vllm-additional-config '{{"weight_nz_mode":0}}' --check-weight-update-equal --weight-checker-timeout 300
        """
        repo = Path(__file__).resolve().parents[1]
        with ray.init(address="local", num_gpus=0, resources={"NPU": 3}, include_dashboard=False) as cluster:
            completed = subprocess.run(
                [sys.executable, str(repo / "train.py"), *shlex.split(args)],
                cwd=repo,
                env={**os.environ, "RAY_ADDRESS": cluster.address_info["address"]},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=900,
            )
            print(completed.stdout, flush=True)
            completed.check_returncode()
            assert "Weight checker passed; generation is ready" in completed.stdout, "First-sync check was skipped"


if __name__ == "__main__":
    os.environ.setdefault("VIME_PLATFORM", "npu")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES", "1")
    main()
