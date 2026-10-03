"""Launch recurrent policy training through VIME's standard Ray entry."""

import argparse
import json
import math
import os
import runpy
import sys
from pathlib import Path


def main(advantage_estimator="ppo", model_family: str | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--engine-revision", required=True)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--precision", choices=("fp16", "fp32"))
    parser.add_argument("--advantage-estimator", choices=("ppo", "grpo"), default=advantage_estimator)
    parser.add_argument("--algorithm", choices=("ppo", "grpo", "dppo", "flow-dppo"))
    parser.add_argument("--flow-dppo-divergence-budget", type=float)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--n-samples-per-prompt", type=int)
    options, extra = parser.parse_known_args()
    algorithm = options.algorithm or options.advantage_estimator
    if options.flow_dppo_divergence_budget is not None:
        if algorithm not in ("ppo", "flow-dppo"):
            parser.error("A Flow-DPPO divergence budget requires the Flow-DPPO policy loss")
        algorithm = "flow-dppo"
    grpo = algorithm == "grpo"
    group_size = options.n_samples_per_prompt
    if group_size is None:
        group_size = 4 if grpo else 1
    if options.rollout_batch_size < 1 or group_size < (2 if grpo else 1):
        parser.error("Use a positive prompt count and at least two samples per prompt for GRPO")
    batch_size = options.rollout_batch_size * group_size
    if algorithm == "flow-dppo" and batch_size % 2:
        parser.error("Flow-DPPO's two updates require an even completion count")
    config = json.loads((options.model / "config.json").read_text())
    family = config["model_type"]
    if model_family is not None and family != model_family:
        parser.error(f"This recipe requires a {model_family} checkpoint, received {family}")
    providers = {"ouro": "ouro", "nanbeige": "nanbeige", "huginn_raven": "huginn"}
    provider = providers[family]
    precision = options.precision or ("fp16" if family == "huginn_raven" else "fp32")
    depth = config[{"ouro": "total_ut_steps", "nanbeige": "num_loops", "huginn_raven": "mean_recurrence"}[family]]
    if family == "huginn_raven":
        layers, width, heads = config["n_layers"], config["n_embd"], config["n_heads"]
        vocab, context = config["padded_vocab_size"], config["block_size"]
        kv_heads, rope = heads, config["rope_base"]
        norm_eps, tied = config["norm_eps"], config["tie_embeddings"]
    else:
        layers, width, heads = config["num_hidden_layers"], config["hidden_size"], config["num_attention_heads"]
        vocab, context = config["vocab_size"], config["max_position_embeddings"]
        kv_heads, rope = config["num_key_value_heads"], config["rope_theta"]
        norm_eps, tied = config["rms_norm_eps"], config["tie_word_embeddings"]
    options.output.mkdir(parents=True, exist_ok=True)
    checkpoints = options.output / "checkpoints"
    next_update = (
        int((checkpoints / "actor/latest_checkpointed_iteration.txt").read_text()) + 1 if options.resume else 0
    )
    roles = {
        "megatron": [
            {
                "role": role,
                "overrides": {
                    "load": str(checkpoints / role if options.resume else options.model),
                    "save": str(checkpoints / role),
                    "use_distributed_optimizer": False,
                },
            }
            for role in (("actor",) if grpo else ("actor", "critic"))
        ]
    }
    roles_path = options.output / "roles.json"
    roles_path.write_text(json.dumps(roles, indent=2))
    flags = {
        "rollout-backend": "vllm-rlt",
        "custom-model-provider-path": f"vime_plugins.{provider}.model.model_provider",
        "rollout-function-path": "vime.rollout.vllm_rlt_rollout.generate_rollout",
        "hf-checkpoint": options.model,
        "load": checkpoints / "actor" if options.resume else options.model,
        "save": checkpoints / "actor",
        "megatron-config-path": roles_path,
        "rlt-model-revision": options.model_revision,
        "rlt-engine-revision": options.engine_revision,
        "rlt-start-version": next_update,
        "actor-num-nodes": 1,
        "actor-num-gpus-per-node": 1,
        "num-gpus-per-node": 2,
        "rollout-num-gpus": 1,
        "rollout-num-gpus-per-engine": 1,
        "num-layers": layers,
        "hidden-size": width,
        "num-attention-heads": heads,
        "ffn-hidden-size": config["intermediate_size"],
        "kv-channels": config["head_dim"],
        "num-query-groups": kv_heads,
        "vocab-size": vocab,
        "normalization": "RMSNorm",
        "norm-epsilon": norm_eps,
        "position-embedding-type": "rope",
        "rotary-base": int(rope),
        "seq-length": 256,
        "max-position-embeddings": context,
        "prompt-data": options.data,
        "input-key": "prompt",
        "label-key": "label",
        "rm-type": "math",
        "num-rollout": options.updates,
        "rollout-batch-size": options.rollout_batch_size,
        "n-samples-per-prompt": group_size,
        "global-batch-size": batch_size // 2 if algorithm == "flow-dppo" else batch_size,
        "micro-batch-size": 1,
        "num-steps-per-rollout": 1,
        "rollout-max-response-len": 48,
        "rollout-max-prompt-len": 128,
        "rollout-temperature": 0.8,
        "rollout-top-p": 1,
        "rlt-kv-blocks": 2 * math.ceil((128 + 48) / 16) * depth,
        "rlt-max-num-seqs": 2,
        "advantage-estimator": "grpo" if grpo else "ppo",
        "eps-clip": 0.2,
        "kl-coef": 0,
        "entropy-coef": 0,
        "optimizer": "sgd",
        "sgd-momentum": 0,
        "lr": 1e-5,
        "lr-decay-style": "constant",
        "lr-decay-iters": 100,
        "weight-decay": 0,
        "save-interval": options.updates,
        "update-weight-mode": "full",
        "update-weight-transport": "disk",
        "update-weight-disk-dir": options.output / "publications",
        "transformer-impl": "local",
        "attention-dropout": 0,
        "hidden-dropout": 0,
        "seed": 42,
        "tensor-model-parallel-size": 1,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
    }
    switches = [
        "fp16" if precision == "fp16" else "recurrent-fp32",
        "swiglu",
        "group-query-attention",
        "disable-bias-linear",
        "offload-train",
        "accumulate-allreduce-grads-in-fp32",
        "rollout-global-dataset",
        "no-gradient-accumulation-fusion",
        "no-rope-fusion",
        "deterministic-mode",
    ]
    if algorithm == "dppo":
        switches.append("use-tis")
        flags.update({"tis-clip-low": 0.5, "tis-clip": 2.0})
    else:
        switches.append("use-rollout-logprobs")
    if algorithm == "flow-dppo":
        flags.update(
            {
                "flow-dppo-divergence-budget": (
                    0.01 if options.flow_dppo_divergence_budget is None else options.flow_dppo_divergence_budget
                ),
                "num-steps-per-rollout": 2,
            }
        )
    if not grpo:
        flags.update({"value-clip": 0.2, "gamma": 1, "lambd": 0.95})
        switches.append("normalize-advantages")
    if precision == "fp16":
        flags["loss-scale"] = 128
    if not tied:
        switches.append("untie-embeddings-and-output-weights")
    if options.recompute:
        flags.update({"recompute-granularity": "full", "recompute-method": "uniform", "recompute-num-layers": 1})
    if options.stop_after is not None:
        flags["stop-after-rollout"] = options.stop_after
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "train.py")]
    for name, value in flags.items():
        command.extend((f"--{name}", str(value)))
    command.extend(f"--{name}" for name in switches)
    command.extend(extra)
    import ray

    source_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, source_root)
    runtime_env_vars = {
        "PYTHONPATH": source_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NCCL_ALGO": "Ring",
    }
    os.environ.update(runtime_env_vars)
    ray.init(
        address=options.ray_address,
        runtime_env={
            "py_executable": sys.executable,
            "env_vars": runtime_env_vars,
        },
    )
    sys.argv = command[1:]
    runpy.run_path(command[1], run_name="__main__")


if __name__ == "__main__":
    main()
