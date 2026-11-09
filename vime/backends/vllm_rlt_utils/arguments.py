"""Arguments for the pinned public vLLM-RLT training contract."""

import json
from pathlib import Path


def add_arguments(parser):
    group = parser.add_argument_group("vLLM-RLT")
    group.add_argument("--rlt-depth", type=int)
    group.add_argument("--rlt-kv-blocks", type=int, default=128)
    group.add_argument("--rlt-max-num-seqs", type=int, default=8)
    group.add_argument("--rlt-attention-backend", choices=("torch", "triton"), default="triton")
    group.add_argument("--rlt-cuda-graphs", action="store_true")
    group.add_argument(
        "--rlt-start-version", type=int, default=0, help="Last published policy version before a fresh resume"
    )
    group.add_argument("--rlt-model-revision", required=True)
    group.add_argument("--rlt-engine-revision", required=True)
    return parser


def validate_args(args):
    config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    family = config["model_type"].lower()
    depth_keys = {"ouro": "total_ut_steps", "nanbeige": "num_loops", "huginn_raven": "mean_recurrence"}
    if family not in depth_keys:
        raise ValueError(f"vLLM-RLT does not support {family!r}")
    args.rlt_model_family = family
    full_depth = config[depth_keys[family]]
    if args.rlt_depth is None:
        args.rlt_depth = full_depth
    if args.rlt_depth != full_depth:
        raise ValueError("Native rollout currently requires the checkpoint's fixed full depth")
    if min(args.rlt_kv_blocks, args.rlt_max_num_seqs) < 1:
        raise ValueError("RLT cache blocks and sequence count must be positive")
    if args.rlt_start_version < 0:
        raise ValueError("RLT's starting policy version must be nonnegative")
    if args.rollout_num_gpus != 1 or args.rollout_num_gpus_per_engine != 1:
        raise ValueError("Native RLT currently uses one rollout GPU and one engine")
    if args.colocate or args.offload_rollout or args.release_train or args.use_fault_tolerance:
        raise ValueError("Native RLT requires dedicated rollout resources without offload or recovery")
    if args.rollout_external or args.use_opd or args.use_rollout_routing_replay or args.check_weight_update_equal:
        raise ValueError("Native RLT does not expose external HTTP, OPD, MoE replay or weight-check RPCs")
    if args.update_weight_mode != "full" or args.update_weight_transport != "disk":
        raise ValueError("Native RLT uses full physical-weight publication via disk")
    if args.update_weight_local_checkpoint_dir:
        raise ValueError("Native RLT reads the publication directory directly")
    if args.rollout_top_p != 1.0 or args.rollout_top_k != -1:
        raise ValueError("Native RLT training requires full-vocabulary sampling")
    if args.rollout_function_path != "vime.rollout.vllm_rlt_rollout.generate_rollout":
        raise ValueError("Use the native RLT rollout function with --rollout-backend=vllm-rlt")
    if args.eval_interval is not None:
        raise ValueError("Native RLT evaluation is not yet exposed through the dataset evaluator")
