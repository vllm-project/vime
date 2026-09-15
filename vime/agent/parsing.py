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
        # Repair before parsing, not after: vLLM logs its own exception on a
        # failed parse, so post-hoc recovery leaves the error in the log.
        body_text = _presanitize_tool_calls(body_text)

        info = None
        try:
            info = parser.extract_tool_calls(body_text, request)
        except Exception:
            logger.warning("[agent.parsing] vllm tool-call parsing raised; trying lenient re-parse")

        # vLLM's Hermes parser does not raise on bad JSON -- it logs and returns
        # tools_called=False (hermes_tool_parser.py:115). So "no tool calls" plus
        # a <tool_call> marker still present in the text means the parser failed,
        # not that the model declined to call a tool. That is the path to recover
        # on; the except branch above only catches parsers that do raise.
        if (info is None or not info.tools_called) and _HERMES_RE.search(body_text):
            valid = {t.get("function", {}).get("name") for t in tools_schema}
            recovered = _lenient_hermes_tool_calls(body_text, {v for v in valid if v})
            if recovered:
                logger.warning("[agent.parsing] lenient re-parse recovered %d tool call(s)", len(recovered))
                return _HERMES_RE.sub("", body_text).strip(), recovered, True
            logger.warning("[agent.parsing] lenient re-parse found nothing; dropping tool call")

        if info is not None and info.tools_called:
            body_text = info.content or ""
            for call in info.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}", strict=False)
                except json.JSONDecodeError:
                    args = {"_raw_arguments": call.function.arguments}
                    ill_formed = True
                tool_uses.append({"name": call.function.name or "tool", "input": args})

    if not tool_uses and tools_schema:
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses, ill_formed


_HERMES_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ESCAPE_FIX_RE = re.compile(r'\\(?!["\\\\/bfnrtu])')


def _presanitize_tool_calls(body_text: str) -> str:
    """Rewrite <tool_call> blocks so vLLM's strict parser accepts them.

    Recovering *after* the parser fails still leaves vLLM's own
    logger.exception in the log, so the failure is only papered over. Repairing
    the text first means the strict parse succeeds and no error is raised at
    all. Blocks that cannot be repaired are left exactly as they were, so the
    parser sees unchanged input and behaves as before.
    """

    def _fix(m: "re.Match[str]") -> str:
        raw = m.group(1)
        for candidate in (raw, _ESCAPE_FIX_RE.sub(r"\\\\", raw)):
            try:
                obj = json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                continue
            args = obj.get("arguments")
            if isinstance(args, str):
                try:
                    obj["arguments"] = json.loads(args, strict=False)
                except json.JSONDecodeError:
                    pass
            return f"<tool_call>{json.dumps(obj, ensure_ascii=False)}</tool_call>"
        return m.group(0)

    return _HERMES_RE.sub(_fix, body_text)


def _lenient_hermes_tool_calls(body_text: str, valid_names: set[str]) -> list[dict[str, Any]]:
    """Recover Hermes tool calls that vLLM's strict parser rejected.

    The model routinely writes source code into an argument string without
    escaping newlines or backslashes, which trips json.loads' strict mode
    ("Invalid control character", "Invalid \\escape"). Those calls are well
    formed apart from the escaping, so re-parse with strict=False rather than
    dropping the whole turn.
    """
    out: list[dict[str, Any]] = []
    for m in _HERMES_RE.finditer(body_text):
        raw = m.group(1)
        try:
            obj = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            # Second pass: a lone backslash that starts no valid JSON escape is
            # the other common way a model mangles a Windows path or a regex.
            # Doubling it is safe -- valid escapes are left untouched.
            try:
                obj = json.loads(_ESCAPE_FIX_RE.sub(r"\\\\", raw), strict=False)
            except json.JSONDecodeError:
                continue
        name = obj.get("name")
        if not name or (valid_names and name not in valid_names):
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except json.JSONDecodeError:
                args = {"_raw_arguments": args}
        out.append({"name": name, "input": args if isinstance(args, dict) else {"_raw_arguments": args}})
    return out


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
