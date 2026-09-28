"""Model-output parsing helpers for agent harnesses."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of one decoded model output."""

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]
    ill_formed: bool = False


def parse_model_output(
    raw_output: str,
    *,
    tokenizer=None,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> ParsedModelOutput:
    """Parse raw model text into reasoning, visible text, and tool uses.

    The heavy format-specific work is delegated to vLLM's reasoning and
    tool-call parsers. The XML fallback covers Anthropic-style tool-call
    text that some coding-agent models still emit occasionally.
    """
    reasoning, body_text = "", raw_output
    if reasoning_parser_name:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.reasoning import ReasoningParserManager

        parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(tokenizer)
        r, b = parser.extract_reasoning(raw_output, ChatCompletionRequest(messages=[]))
        reasoning, body_text = r or "", b or ""
        if not reasoning and "</think>" in body_text:
            reasoning, body_text = body_text.split("</think>", 1)

    body_text, tool_uses, ill_formed = parse_tool_uses(body_text, tools_schema, tool_parser_name, tokenizer)
    return ParsedModelOutput(
        reasoning=reasoning,
        text=(body_text or "").strip(),
        tool_uses=tool_uses,
        ill_formed=ill_formed,
    )


def parse_tool_uses(
    body_text: str,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    tokenizer,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Parse tool calls from body text and return visible text plus tool uses."""
    tool_uses: list[dict[str, Any]] = []
    ill_formed = False
    if tool_parser_name and tools_schema:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.tool_parsers import ToolParserManager

        request = ChatCompletionRequest(messages=[], tools=tools_schema)
        parser = ToolParserManager.get_tool_parser(tool_parser_name)(tokenizer, tools=request.tools)
        info = None
        try:
            info = parser.extract_tool_calls(body_text, request)
        except Exception:
            logger.exception("[agent.parsing] vllm tool-call parsing failed; falling back")
        if info is not None and info.tools_called:
            body_text = info.content or ""
            for call in info.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw_arguments": call.function.arguments}
                    ill_formed = True
                tool_uses.append({"name": call.function.name or "tool", "input": args})

    if not tool_uses and tools_schema:
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses, ill_formed


def parse_xml_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Anthropic-style XML tool calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last = 0
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        body_text,
        flags=re.DOTALL,
    ):
        name, inner = m.group(1), m.group(2)
        if name in valid_tools:
            args = {
                p.group(1): p.group(2).strip()
                for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
            }
            tool_uses.append({"name": name, "input": args})
            cleaned_parts.append(body_text[last : m.start()])
            last = m.end()
    cleaned_parts.append(body_text[last:])
    return "".join(cleaned_parts), tool_uses
