import os

from vime.utils.external_utils.command_utils import execute_train_npu

MODEL_NAME = os.environ.get("VIME_SCRIPT_MODEL_NAME", "Qwen3-VL-8B-Instruct")
SUPPORTED_MODELS = {
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Thinking",
}
if MODEL_NAME not in SUPPORTED_MODELS:
    raise ValueError(f"Unsupported VIME_SCRIPT_MODEL_NAME={MODEL_NAME}")

NUM_ROLLOUT = int(os.environ.get("VIME_SCRIPT_NUM_ROLLOUT", "3000"))
if NUM_ROLLOUT <= 0:
    raise ValueError("VIME_SCRIPT_NUM_ROLLOUT must be positive")

MODEL_ROOT = os.environ.get("VIME_SCRIPT_MODEL_ROOT", "/root/models")
DATA_ROOT = os.environ.get(
    "VIME_SCRIPT_DATA_ROOT", "/root/datasets/geo3k_imgurl_processed"
)
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")


def get_megatron_model_type(model_name: str) -> str:
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")


def execute():
    model_path = os.path.join(MODEL_ROOT, MODEL_NAME)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {TRAIN_DATA_PATH}")

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project vime-dev "
            "--wandb-group geo3k_vlm_multi_turn "
            f"--wandb-key '{wandb_api_key}' "
        )
        if wandb_api_key
        else ""
    )

    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--rm-type math "
        "--custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate "
        "--custom-config-path examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml "
        "--rollout-shuffle "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-kl-loss "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--vllm-gpu-memory-utilization 0.6 "
        "--vllm-max-model-len 16384 "
        "--vllm-generation-config auto "
        "--vllm-logprobs-mode processed_logprobs "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load {model_path} "
        f"--ref-load {model_path} "
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--balance-data "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
        "--no-gradient-accumulation-fusion "
        "--use-flash-attn "
    )

    megatron_model_type = get_megatron_model_type(MODEL_NAME)
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"

    train_args = (
        f"--hf-checkpoint {model_path} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{vllm_args} "
        f"{megatron_args} "
        f"{misc_args} "
        f"{wandb_args} "
    )

    execute_train_npu(
        train_args=train_args,
        megatron_model_type=megatron_model_type,
        extra_env_vars={"WANDB_API_KEY": wandb_api_key} if wandb_api_key else {},
    )


if __name__ == "__main__":
    execute()
