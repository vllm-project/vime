# Tau bench
This example shows vime training in an agentic multi-turn tool use environment.


## Environment Setup
This example assumes a vime container image. Install tau-bench dependencies:

```bash
cd /root/
git clone https://github.com/JD-ETH/tau-bench.git
cd tau-bench
git checkout feature/litellm-retry
pip install -e . --no-deps
pip install litellm
```

Use the following script to generate task index jsonl for training:

```bash
cd /root/vime/examples/tau-bench
python tau1_mock.py --local_dir /root/tau-bench/
```

Initialize the Qwen3-4B-Instruct-2507 model needed for tool use:

```bash
# hf checkpoint
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# mcore checkpoint
cd /root/vime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

## Running the Script

You need to configure your litellm API in generate_with_tau.py for user simulation:

TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"], only tool-calling implemented for now
    "user_model": "gemini-2.0-flash-lite",  # Cheap Model for user simulator
    "user_model_provider": "gemini",
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail, ["test"] for airline
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router", # Unused, required
    "model": "qwen3-4b", # Unused, reqired
}
# Replace with your actual API key for user sim    
GEMINI_API_KEY = "YOUR KEY" 

Multi-turn limit: set env `TAU_MAX_TURNS` (default 10) or pass `--max-turns` to train.py.

Agent rollout always uses vLLM (`/inference/v1/generate`); only `TAU_CONFIGS` controls the user simulator.

And run:

```bash
cd /root/vime
bash examples/tau-bench/run_qwen3_4B.sh
```
