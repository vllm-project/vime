# Tau-Bench Multi-Turn Tool Use

This example shows vime training in an agentic multi-turn tool use environment based on [tau-bench](https://github.com/JD-ETH/tau-bench).

## Environment Setup

Use the `vime_v22` container image and initialize the environment:

```bash
cd /data/nfs_87/xky/vime_debug/vime
pip install -e . --no-deps

cd /data/nfs_87/xky/
git clone https://github.com/JD-ETH/tau-bench.git
cd tau-bench
git checkout feature/litellem-retry
pip install -e . --no-deps
```

Generate mock data for training:

```bash
cd /data/nfs_87/xky/vime_debug/vime/examples/tau_bench
python tau1_mock.py --local_dir /data/nfs_87/xky/datasets/tau-bench/
```

Initialize the Qwen3-4B model:

```bash
# HF checkpoint (if not already available)
hf download Qwen/Qwen3-4B --local-dir /data/nfs_87/xky/models/Qwen3-4B

# MCore checkpoint
cd /data/nfs_87/xky/vime_debug/vime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /data/nfs_87/xky/models/Qwen3-4B \
    --save /data/nfs_87/xky/models/Qwen3-4B_torch_dist
```

## Running the Script

Configure your litellm API in `generate_with_tau.py` for user simulation:

```python
TAU_CONFIGS = {
    "env": "retail",
    "agent": "tool-calling",
    "user_model": "gemini-2.0-flash-lite",
    "user_model_provider": "gemini",
    "task_split": "train",
    "user_strategy": "llm",
    "model_provider": "auto_router",
    "model": "qwen3-4b",
}
GEMINI_API_KEY = "YOUR KEY"
```

Then run:

```bash
cd /data/nfs_87/xky/vime_debug/vime
bash examples/tau_bench/run_qwen3_4B.sh
```

## Architecture

This example adapts the slime tau-bench example to vime's multi-turn rollout architecture:

- `rollout.py`: Custom multi-turn generate function using vLLM render route, following the pattern from `geo3k_vlm_multi_turn/rollout.py` with PR #120 bug fixes (proper token budget tracking, EOS after stop string, prefix stability validation).
- `env_tau.py`: Tau-bench interaction environment wrapping the tau-bench toolkit, implementing the `BaseInteractionEnv` interface.
- `generate_with_tau.py`: Tau-bench configuration and entry point for the generate function.
- `openai_tool_adapter.py`: Adapter converting sglang tool call parsing to OpenAI-compatible format.
- `sglang_tool_parser.py`: Tool call parser using sglang's FunctionCallParser.
- `tau1_mock.py`: Mock data generation script for tau-bench tasks.

## Key Differences from slime Version

1. Uses vLLM rollout backend instead of SGLang
2. Uses vLLM render route (`/v1/chat/completions/render` + `/inference/v1/generate`) for multi-turn generation
3. Proper token budget tracking across turns (PR #120 fix)
4. Appends EOS token after stop strings in multi-turn (PR #120 fix)
5. Validates prefix stability between render turns (PR #120 fix)
6. Uses vime's `Sample` type and `GenerateState` utilities
