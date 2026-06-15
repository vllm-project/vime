"""Retool-style code-execution ``agent_step`` for vime's multi-turn agent scaffold.

Reproduces the maestro agent loop of ai21-verl's
``exp_regression_maestro_retool.py``: a math policy that solves problems by writing
Python, executing it, reading stdout, and iterating until it emits a final
``\\boxed{}`` answer.

How maestro ran it (and why this is faithful, not a re-architecture):
    Maestro exposed an ``execute_python_code`` HTTP tool whose endpoint was
    ``CodeRunnerService.get_url()`` == ``http://<deployment>/run_code`` (an
    ``eval-code-runner`` Helm deployment). Crucially the agent loop reached it
    *only by URL* — the maestro job builder injects that URL as the
    ``CODE_RUNNER_URL`` env var. So the runner is just an HTTP service that can
    live anywhere: its own autoscaled k8s deployment (prod) **or** a process in
    your debug pod (``CODE_RUNNER_URL=http://localhost:8000/run_code``). This
    module POSTs the same ``{"code": ...}`` body to ``CODE_RUNNER_URL`` and feeds
    the returned stdout back as the (loss-masked) observation.

Wiring (see scripts/run-qwen2.5-7B-ai21-maestro-retool-4gpu.sh)::

    --custom-generate-function-path vime_plugins.rollout.multi_turn_agent.generate
    --custom-config-path scripts/maestro-retool-agent.yaml   # agent_step_path, agent_max_turns, retool_*
    # env: CODE_RUNNER_URL=http://localhost:8000/run_code

The ``multi_turn_agent`` scaffold owns token accounting / loss masking / status /
budgets. This ``agent_step`` only decides, per turn:

    model called the tool  -> run the code, return stdout as a masked observation, keep looping
    no tool call           -> treat as the final answer turn, stop

Reward is left to ``async_rm`` (``--custom-rm-path`` ai21 evaluators, same as the
other regression scripts) — this module never sets ``reward``.

IMPORTANT — observation wrapping must match the SFT model's training format.
``qwen-25-7b-retool-sft`` was trained on one specific transcript shape. Maestro
defined the tool as OpenAI-style function calling, so the default here wraps the
result as a Qwen ``<tool_response>`` turn and re-opens an assistant turn. If the
model was actually trained on inline ReTool ``<code>``/``<interpreter>`` blocks,
set ``retool_observation_format: interpreter`` in the YAML. ``auto`` (the default)
picks the wrapping that matches how the *call* was parsed.
"""

import json
import re
from typing import Any

from vime_plugins.utils.common import cfg as _cfg

# ---- code-call extraction ---------------------------------------------------
# OpenAI/Qwen function-call envelope: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Inline ReTool blocks: ```python ... ```  /  ```py ... ```  /  ```...```  /  <code>...</code>
_FENCED_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_CODE_TAG_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{")

_TOOL_NAME = "execute_python_code"


def _extract_code(text: str) -> tuple[str | None, str]:
    """Return (code, call_style). call_style is 'tool_call' | 'inline' | '' (none)."""
    # 1) OpenAI/Qwen tool-call JSON (maestro's literal http_tools schema).
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        fn = payload.get("function", payload)
        if fn.get("name") and fn.get("name") != _TOOL_NAME:
            continue
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"code": arguments}
        code = arguments.get("code")
        if code:
            return code, "tool_call"
    # 2) Inline ReTool fenced / <code> blocks (use the last one in the turn).
    blocks = _FENCED_RE.findall(text) or _CODE_TAG_RE.findall(text)
    if blocks:
        return blocks[-1].strip(), "inline"
    return None, ""


async def _run_code(code: str, url: str, timeout: float) -> str:
    """POST {"code": ...} to the code runner and return its stdout (best-effort)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json={"code": code})
            resp.raise_for_status()
            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError):
                return resp.text
    except Exception as exc:  # noqa: BLE001 — surface any runner error to the model, don't crash the rollout
        return f"[code-runner error] {type(exc).__name__}: {exc}"
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        # eval-code-runner / common runners expose one of these; concatenate stdout+stderr.
        out = data.get("stdout") or data.get("output") or data.get("result") or ""
        err = data.get("stderr") or data.get("error") or ""
        return f"{out}{err}".strip() or json.dumps(data)
    return str(data)


def _wrap_observation(result: str, fmt: str, call_style: str) -> str:
    """Format the tool result the way the SFT model expects to read it next turn."""
    if fmt == "auto":
        fmt = "tool_response" if call_style == "tool_call" else "interpreter"
    if fmt == "interpreter":
        # Inline ReTool: stays inside the same assistant turn, model keeps generating.
        return f"\n<interpreter>\n{result}\n</interpreter>\n"
    # Qwen tool-response role switch, then re-open an assistant turn. The Qwen tokenizer
    # maps these literal <|im_*|> markers to their special-token ids (add_special_tokens=False
    # in the scaffold), so this reproduces the chat-template framing without re-rendering.
    return (
        f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n</tool_response>"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )


async def agent_step(args, sample, assistant_text, turn_index) -> dict[str, Any]:
    code, call_style = _extract_code(assistant_text)

    # No tool call -> this was the final-answer turn. Stop; reward comes from async_rm.
    if not code:
        return {"done": True, "observation": None, "reward": None}

    url = _cfg(args, "code_runner_url", "CODE_RUNNER_URL", "http://localhost:8000/run_code")
    timeout = float(_cfg(args, "retool_tool_timeout", "RETOOL_TOOL_TIMEOUT", 10))
    fmt = str(_cfg(args, "retool_observation_format", "RETOOL_OBSERVATION_FORMAT", "auto"))

    result = await _run_code(code, url, timeout)
    observation = _wrap_observation(result, fmt, call_style)

    # If the model already emitted a final boxed answer alongside the code, stop after this turn.
    done = bool(_BOXED_RE.search(assistant_text))
    return {"done": done, "observation": observation, "reward": None}
