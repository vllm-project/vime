source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/qwen3-30B-A3B.sh"

MODEL_ARGS+=(
   --spec vime_plugins.models.qwen3_omni_moe get_qwen3_omni_model_provider
   --use-qwen-vl
)
