# Retool: from SFT to RL

This example demonstrates how to use the retool functionality for tool-enabled language model generation.

## Overview

The retool example provides:
- Safe Python code execution in a sandbox environment
- Tool registry for managing available tools
- Integration with language model generation
- Reward calculation for tool usage

## Files

- `generate_with_retool.py`: Main generation function with tool support
- `tool_sandbox.py`: Tool execution and safety management
- `sft_data_processing.py`: Process SFT dataset
- `rl_data_preprocess.py`: Process the RL (DAPO-Math-17k) dataset

## Usage

1. Setup and download datasets:
```bash
cd vime
pip install -e . --no-deps
pip install -r examples/retool/requirements.txt
# For SFT part, you can use later model to RL directly and skip SFT.
hf download --repo-type dataset JoeYing/ReTool-SFT  --local-dir /root/JoeYing/ReTool-SFT
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen/Qwen3-4B-Instruct-2507

# For RL part
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /root/aime-2024
# download our SFT model if you want to skip SFT
hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir /root/font-info/qwen3-4b-sft
```

2. Create torch dist

Both checkpoints use rope theta `5e6`, which differs from the `1e6` default in
`scripts/models/qwen3-4B.sh`. Override it with `MODEL_ARGS_ROTARY_BASE` so the
conversion and the training scripts agree.

For SFT
```bash
MODEL_ARGS_ROTARY_BASE=5000000 source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen/Qwen3-4B-Instruct-2507_torch_dist
```

Or RL only
```bash
MODEL_ARGS_ROTARY_BASE=5000000 source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/font-info/qwen3-4b-sft \
    --save /root/font-info/qwen3-4b-sft_torch_dist
```

3. SFT:
```bash
python examples/retool/sft_data_processing.py
bash examples/retool/retool_qwen3_4b_sft.sh
```

4. RL:
```bash
bash examples/retool/retool_qwen3_4b_rl.sh
```

5. Use in your training scripts by importing the generate function:
```python
from generate_with_retool import generate, reward_func
```

The RL script wires these up with:
```bash
--custom-generate-function-path generate_with_retool.generate
--custom-rm-path generate_with_retool.reward_func
```
`generate_with_retool` is resolved as a top-level module, so the example
directory is added to `PYTHONPATH` in the script's Ray runtime env (this is also
what lets it import its sibling `tool_sandbox`).

## Tool Format

The system uses the following tool format:

```
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute."}}, "required": ["code"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

## Safety Features

- Code execution in isolated sandbox
- Memory and time limits
- Dangerous operation detection
- Allowed module restrictions

Note that `PythonSandbox._check_code_safety` is deliberately strict: it allows
only the stdlib modules in `PythonSandbox.allowed_modules` (`math`, `random`,
`statistics`, `decimal`, `fractions`, …) and rejects `eval`/`exec`/`open`,
dunder access, and imports outside that set. Widen `allowed_modules` if your task
needs more (e.g. `sympy` or `numpy`).

## Notes on the vLLM port

This example was ported from slime's SGLang implementation. The rollout loop
talks to vime's vLLM router at `/inference/v1/generate` with a
`{"model", "token_ids", "sampling_params"}` body, and reads back
`choices[0].token_ids` plus `choices[0].logprobs.content[i].logprob`.

Two behaviours differ from `vime.rollout.vllm_rollout.generate` on purpose:

- When the engine returns tokens but no usable per-token logprobs, this example
  marks the sample `ABORTED` instead of substituting zeros. Zero-filled logprobs
  would desync `rollout_log_probs` from the response tokens and silently corrupt
  the importance ratio, so the sample is returned to the buffer for retry.
- The tool-concurrency limit is taken exactly once, inside
  `ToolRegistry.execute_tool`. `tool_sandbox.SEMAPHORE` is a plain
  `asyncio.Semaphore` and is not reentrant, so acquiring it in both the caller
  and the registry needs two permits per tool call and hangs once enough calls
  are in flight.
