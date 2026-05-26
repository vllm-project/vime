"""Hermes / Qwen 2.5 tool-call parser, self-contained.

Mirrors vLLM's `Hermes2ProToolParser.extract_tool_calls`
(see `vllm/tool_parsers/hermes_tool_parser.py`) for the non-streaming
case. Qwen 2.5 emits tool calls in the Hermes format:

    <tool_call>{"name": "fn", "arguments": {"k": "v"}}</tool_call>

Multiple <tool_call> blocks per response are supported.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
_SUPPORTED_PARSERS = frozenset({"qwen25", "hermes"})


def parse_tools(
    response: str,
    tools: list[dict[str, Any]],
    parser: str = "qwen25",
) -> dict[str, Any]:
    """Parse Hermes-style `<tool_call>{...}</tool_call>` blocks from an LLM response.

    Args:
        response: Raw model output text.
        tools: Tool specs (accepted for API compatibility; the extraction is
            content-driven and does not need the tool list).
        parser: Parser id. Only `"qwen25"` and `"hermes"` are supported — both
            consume the `<tool_call>...</tool_call>` JSON-payload format used
            by Qwen 2.5 / Hermes-2 Pro / similar Hermes-family chat templates.

    Returns:
        Dict in the format::

            {
                "normal_text": str,   # text preceding the first <tool_call> tag
                                      # (or the full response if no tool calls)
                "calls": [
                    {"name": str, "parameters": str (JSON-encoded arguments)},
                    ...
                ],
            }
    """
    del tools  # tool list is unused for extraction; kept for API parity

    if parser not in _SUPPORTED_PARSERS:
        raise ValueError(f"Unsupported tool-call parser: {parser!r}. " f"Supported: {sorted(_SUPPORTED_PARSERS)}.")

    if "<tool_call>" not in response:
        return {"normal_text": response, "calls": []}

    calls: list[dict[str, str]] = []
    for match in _TOOL_CALL_RE.findall(response):
        raw_json = match[0] or match[1]
        if not raw_json:
            continue
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            # Skip malformed payloads — matches vLLM Hermes parser behaviour
            # (it logs+swallows in extract_tool_calls).
            continue
        name = obj.get("name", "")
        arguments = obj.get("arguments", {})
        calls.append(
            {
                "name": name,
                "parameters": json.dumps(arguments, ensure_ascii=False),
            }
        )

    normal_text = response[: response.find("<tool_call>")]
    return {"normal_text": normal_text, "calls": calls}
