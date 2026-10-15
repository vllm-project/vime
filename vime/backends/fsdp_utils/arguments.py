"""Flags owned by the FSDP2 training backend.

Only options that the Megatron/vime parsers do not already define live here.
Optimizer, LR schedule, batch, seed, precision and checkpoint settings are read
from the shared namespace (``--lr``, ``--adam-beta1``, ``--clip-grad``,
``--global-batch-size``, ``--save`` ...), so the same command line trains the
same recipe on either backend.
"""

import argparse

_PARAM_DTYPES = ("auto", "bf16", "fp16", "fp32")


def add_fsdp_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("FSDP2 training backend")
    group.add_argument(
        "--fsdp-attn-implementation",
        type=str,
        default="flash_attention_2",
        help="HF attention implementation for the training model (e.g. flash_attention_2, sdpa, eager).",
    )
    group.add_argument(
        "--fsdp-param-dtype",
        type=str,
        choices=_PARAM_DTYPES,
        default="auto",
        help=(
            "Compute dtype of the sharded model. Master weights and optimizer states stay in fp32. "
            "'auto' follows --fp16 (fp16) and otherwise uses bf16."
        ),
    )
    group.add_argument(
        "--fsdp-reduce-dtype",
        type=str,
        choices=("fp32", "bf16"),
        default="fp32",
        help="Dtype used to reduce-scatter gradients.",
    )
    group.add_argument(
        "--fsdp-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Free unsharded parameters after each layer's forward (ZeRO-3); disable for ZeRO-2 behavior.",
    )
    group.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        default=False,
        help="Keep sharded parameters, gradients and optimizer states on CPU (the optimizer step runs on CPU).",
    )
    group.add_argument(
        "--fsdp-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Recompute decoder layers in backward.",
    )
    return parser


def validate_fsdp_args(args, role: str = "actor", hf_config=None) -> None:
    """Reject configurations the FSDP2 backend does not implement yet."""
    unsupported = []
    if role != "actor" or args.use_critic:
        unsupported.append("critic training (--use-critic)")
    for flag, size in (
        ("--tensor-model-parallel-size", args.tensor_model_parallel_size),
        ("--pipeline-model-parallel-size", args.pipeline_model_parallel_size),
        ("--context-parallel-size", args.context_parallel_size),
        ("--expert-model-parallel-size", args.expert_model_parallel_size),
    ):
        if size > 1:
            unsupported.append(f"{flag} > 1 (FSDP2 shards over data parallelism only)")
    if args.use_opd and args.opd_type == "megatron":
        unsupported.append("--opd-type megatron (use --opd-type vllm)")
    if args.dspark_enabled or args.enable_mtp_training:
        unsupported.append("draft/MTP training")
    if args.use_routing_replay or args.use_rollout_routing_replay:
        unsupported.append("routing replay")
    # Delta sync is validated to require the disk transport.
    if args.update_weight_transport == "disk":
        unsupported.append("disk/delta weight sync (use --update-weight-transport nccl)")
    if args.async_save:
        unsupported.append("--async-save")
    if hf_config is not None:
        if hasattr(hf_config, "vision_config"):
            unsupported.append("multimodal models")
        if getattr(hf_config, "quantization_config", None) is not None:
            unsupported.append("quantized checkpoints")
    if unsupported:
        raise ValueError("The fsdp2 training backend does not support: " + "; ".join(unsupported))
