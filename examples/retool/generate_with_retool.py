# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
# Adapted for vLLM /inference/v1/generate endpoint (disagg API)
import logging
import random
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from vime.rollout.vllm_rollout import GenerateState, _build_inference_sampling_params
from vime.utils.http_utils import post
from vime.utils.types import Sample

# Import reward models
try:
    from vime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# ── Sample-level verbose logging ──────────────────────────────────────────
# Roughly 1/20 samples are logged so the output stays readable.
_LOG_SAMPLE_PROB = 0.05
_LOG_WIDTH = 300

# ── Log probability collection ───────────────────────────────────────────
# When True, collect log probabilities for TIS (Trajectory Importance Sampling).
# When True, we CANNOT postprocess the decoded text because that would break
# token/logp alignment. Instead, we rely on stop strings to ensure the engine
# stops at tool/answer boundaries.
RETURN_LOGPROB = True


def _trunc(s: str, n: int = 300) -> str:
    """Truncate *s* to at most *n* characters for display."""
    if len(s) <= n:
        return s
    return s[:n] + f"…[+{len(s) - n}]"


# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You can write Python code to solve problems. Put your code between <code> and </code> tags. The code will be executed and you will see the output.
When you have the final answer, write it as: Answer: \boxed{your_answer}
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = "You are a helpful assistant that solves " "mathematical problems step by step."

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Check for Answer: \boxed{...} format (only format we need for math_dapo)
    # Use a more robust regex that handles nested braces
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness.

    IMPORTANT: Only used when RETURN_LOGPROB is False. When log prob collection
    is enabled, we cannot postprocess the decoded text because that would break
    token/logp alignment. Instead, we rely on stop strings to ensure the engine
    stops at tool/answer boundaries.
    """
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle Answer: \boxed{...} format (only format we need for math_dapo)
    if "Answer:" in resp and "\\boxed{" in resp:
        # Find the last occurrence of Answer: \boxed{...} with nested braces support
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


async def execute_predictions(prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to execute code, I should put the code between "
            "<code> and </code>. "
            "If I want to give the final answer, I should write "
            "'Answer: ' followed by a boxed expression. Let me try again.\n"
        )
        done = False

    return next_obs, done


# Stop tags: make the inference engine STOP at the tool boundary.
# "Answer:" was previously included, but it caused the engine to stop
# immediately after "Answer:" before the model could generate "\boxed{...}",
# resulting in invalid actions and reward=-1 every time.  The model now
# completes its answer naturally; execute_predictions detects the
# Answer: \boxed{...} pattern and sets done=True, ending the turn cleanly.
_STOP_TAGS = ["</code>"]


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls (vLLM version)"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."

    state = GenerateState(args)
    router_ip = args.vllm_router_ip
    router_port = args.vllm_router_port
    url = f"http://{router_ip}:{router_port}/inference/v1/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()

    # Extract the raw user content from sample.prompt to avoid duplicate
    # <|im_start|>user tags — the Jinja2 template adds them itself.
    _raw_prompt = sample.prompt
    if isinstance(_raw_prompt, list):
        # List of message dicts, e.g. [{"role": "user", "content": "..."}]
        _raw_prompt = "".join(m.get("content", "") for m in _raw_prompt)
    # If the prompt string already contains chat-format tags, strip them so
    # the template can re-add them consistently.
    if isinstance(_raw_prompt, str) and "<|im_start|>" in _raw_prompt:
        _raw_prompt = re.sub(r"<\|im_start\|>(?:system|user|assistant)\n?", "", _raw_prompt)
        _raw_prompt = _raw_prompt.replace("<|im_end|>", "")
        _raw_prompt = _raw_prompt.strip()

    prompt = format_conversation_with_tools(prompt=_raw_prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    rollout_log_probs = [] if RETURN_LOGPROB else None
    tool_call_count = 0  # Track actual tool call rounds
    consecutive_memory_errors = 0  # Track consecutive "Memory usage too high" errors
    obs_truncated = False  # Flag: obs caused total length to exceed max_context_length
    # Calculate max context length once at the beginning
    max_context_length = len(prompt_tokens_ids) + args.rollout_max_response_len
    logger.debug("max_context_length is set to %d", max_context_length)

    # Randomly select a small fraction of samples for detailed turn-by-turn logging.
    verbose = random.random() < _LOG_SAMPLE_PROB
    if verbose:
        _sep = "═" * _LOG_WIDTH
        _prompt_display = (
            "".join(m.get("content", "") for m in sample.prompt) if isinstance(sample.prompt, list) else sample.prompt
        )
        logger.debug(
            "\n%s\n[ReTool LOG] prompt (%d tokens): %s\n%s",
            _sep,
            len(prompt_tokens_ids),
            _trunc(_prompt_display, 200),
            _sep,
        )

    # Add stop tags to sampling_params so the engine stops at tool/answer boundaries.
    # This is critical for keeping token/logp alignment when RETURN_LOGPROB is enabled.
    _existing_stop = sampling_params.get("stop") or []
    if isinstance(_existing_stop, str):
        _existing_stop = [_existing_stop]
    sampling_params = {**sampling_params, "stop": list(dict.fromkeys([*_existing_stop, *_STOP_TAGS]))}

    # Build vLLM-style sampling params (maps max_new_tokens -> max_tokens, adds logprobs, etc.)
    inference_sampling_params = _build_inference_sampling_params(sampling_params)

    for turn in range(TOOL_CONFIGS["max_turns"]):
        # vLLM /inference/v1/generate requires token_ids instead of text.
        # For multi-turn, re-tokenize the full context each time.
        full_text = prompt + response
        current_token_ids = state.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Check if total length exceeds max context length
        total_length = len(current_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Dynamically calculate remaining token budget for this turn
        remaining_tokens = max_context_length - total_length

        # Update max_tokens for this turn to respect the remaining budget
        current_inference_params = dict(inference_sampling_params)
        current_inference_params["max_tokens"] = min(
            inference_sampling_params["max_tokens"],
            remaining_tokens,
        )

        # Check if we have budget for more tokens
        if current_inference_params["max_tokens"] <= 0:
            sample.status = Sample.Status.TRUNCATED
            break

        payload = {
            "token_ids": current_token_ids,
            "sampling_params": current_inference_params,
        }
        if hasattr(args, "hf_checkpoint"):
            payload["model"] = args.hf_checkpoint

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<interpreter>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Parse vLLM GenerateResponse: {"choices": [{"token_ids": [...], "logprobs": ..., "finish_reason": "stop"}]}
        choice = output["choices"][0]
        finish_reason = choice.get("finish_reason", "stop")

        # Handle abort
        if finish_reason == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        # Extract token IDs from vLLM response
        cur_response_token_ids = choice.get("token_ids") or []

        # Decode text from token_ids
        skip_sp = current_inference_params.get("skip_special_tokens")
        skip_decode = True if skip_sp is None else bool(skip_sp)
        cur_response = (
            state.tokenizer.decode(cur_response_token_ids, skip_special_tokens=skip_decode)
            if cur_response_token_ids
            else ""
        )

        # Extract log probs if enabled
        if RETURN_LOGPROB:
            cur_response_log_probs: list[float] = []
            lp = choice.get("logprobs")
            if isinstance(lp, dict):
                content_items = lp.get("content") or []
                cur_response_log_probs = [
                    float(item.get("logprob", 0.0)) if isinstance(item, dict) else 0.0 for item in content_items
                ]
            if not cur_response_log_probs:
                cur_response_log_probs = [0.0] * len(cur_response_token_ids)
        else:
            # When not collecting log probs, we can safely postprocess the response
            cur_response = postprocess_responses(cur_response)
            # Re-tokenize after postprocessing
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
            cur_response_log_probs = None

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Add log probs if enabled
        if RETURN_LOGPROB:
            rollout_log_probs += cur_response_log_probs

        # verbose: show what the model generated this turn
        if verbose:
            n_tok = len(cur_response_token_ids)
            logger.debug(
                "\n%s\n[Turn %d] model output (%d tok, finish=%s):\n  %s",
                "─" * _LOG_WIDTH,
                turn + 1,
                n_tok,
                finish_reason,
                _trunc(cur_response).replace("\n", "\n  "),
            )

        # Check length limit
        if finish_reason == "length":
            if verbose:
                logger.debug("[Turn %d] → length limit reached, stopping.", turn + 1)
            break

        next_obs, done = await execute_predictions(cur_response)

        # Track consecutive memory errors to break the vicious retry cycle.
        # When sandbox keeps returning "Memory usage too high", the model
        # generates another code attempt which also fails, consuming more
        # tokens and memory. Force done=True after N consecutive failures.
        if "Memory usage too high" in next_obs:
            consecutive_memory_errors += 1
            if consecutive_memory_errors >= TOOL_CONFIGS.get("max_consecutive_memory_errors", 3):
                done = True
        else:
            consecutive_memory_errors = 0

        # verbose: show action and observation
        if verbose:
            if done:
                logger.debug("[Turn %d] → answer detected (DONE)", turn + 1)
            elif "<interpreter>" in next_obs:
                obs_display = "  " + _trunc(next_obs, 300).replace("\n", "\n  ")
                logger.debug("[Turn %d] → code executed, observation:\n%s", turn + 1, obs_display)
            else:
                logger.debug("[Turn %d] → invalid action (no recognized code or answer)", turn + 1)

        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens if enabled (they won't be used due to loss_mask=0)
        if RETURN_LOGPROB:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)

            # Verify alignment when collecting log probs
            assert len(response_token_ids) == len(
                rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

        # Truncate if obs pushed total response tokens beyond max_context_length
        max_response_tokens = max_context_length - len(prompt_tokens_ids)
        if len(response_token_ids) > max_response_tokens:
            response_token_ids = response_token_ids[:max_response_tokens]
            loss_masks = loss_masks[:max_response_tokens]
            if RETURN_LOGPROB:
                rollout_log_probs = rollout_log_probs[:max_response_tokens]
            obs_truncated = True
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    if verbose:
        logger.debug(
            "\n%s\n[ReTool LOG] finished | tool_calls=%d | " "response_tokens=%d | finish=%s\n%s",
            "═" * _LOG_WIDTH,
            tool_call_count,
            len(response_token_ids),
            finish_reason,
            "═" * _LOG_WIDTH,
        )

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.prompt = prompt

    # Store log probs if enabled
    if RETURN_LOGPROB:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status
    # vLLM finish_reason is a string: "stop", "length", or "abort"
    match finish_reason:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    if obs_truncated:
        sample.status = Sample.Status.TRUNCATED

    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Truncated samples: neutral reward — not right, not wrong, just incomplete.
    if sample.status == Sample.Status.TRUNCATED:
        return {"score": 0.0, "acc": False, "pred": ""}

    # Build complete solution string.
    # sample.prompt may be a list of message dicts; flatten to plain text.
    if isinstance(sample.prompt, list):
        prompt_str = "".join(m.get("content", "") for m in sample.prompt)
    else:
        prompt_str = sample.prompt
    solution_str = prompt_str + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # Reward shaping:
    #   Correct + no tools    → 1.0  (pure reasoning, best)
    #   Correct + tools       → 1.0 + bonus (tools helped, still good)
    #   Wrong + no tools      → 0.0  (neutral — model didn't even try tools)
    #   Wrong + tools         → small positive (encourages exploration,
    #                           but capped low to avoid reward hacking:
    #                           the model should not prefer tool-calling
    #                           over getting the right answer)
    if result["score"] > 0:
        result["score"] = 1.0 + min(0.2, num_turns * 0.05)
    else:
        result["score"] = min(0.1, num_turns * 0.02)

    if result["pred"] is None:
        result["pred"] = ""

    return result
