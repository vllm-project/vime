"""Local tool-call parser for vLLM rollout."""

import json
import re
from typing import Any


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25") -> dict[str, Any]:
    if parser == "qwen25":
        return _parse_qwen25_tools(response)
    return _parse_qwen25_tools(response)


def _try_parse_json_tool_call(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            name = parsed.get("name", "")
            parameters = parsed.get("arguments", parsed.get("parameters", {}))
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    pass
            return {"name": name, "parameters": parameters}
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _parse_qwen25_tools(response: str) -> dict[str, Any]:
    call_open = chr(60) + "tool_call" + chr(62)
    call_close = chr(60) + "/tool_call" + chr(62)
    call_open_alt = chr(60) + "call" + chr(62)
    call_close_alt = chr(60) + "/call" + chr(62)
    pattern = r"(?:" + call_open + "|" + call_open_alt + r")\s*(.*?)\s*(?:" + call_close + "|" + call_close_alt + ")"
    tool_call_pattern = re.compile(pattern, re.DOTALL)
    matches = tool_call_pattern.findall(response)

    if matches:
        parts = tool_call_pattern.split(response)
        normal_text = parts[0].strip() if parts else ""
        calls = []
        for match in matches:
            match = match.strip()
            parsed_call = _try_parse_json_tool_call(match)
            if parsed_call:
                calls.append(parsed_call)
            else:
                try:
                    json_match = re.search(r"\{.*\}", match, re.DOTALL)
                    if json_match:
                        parsed_call = _try_parse_json_tool_call(json_match.group())
                        if parsed_call:
                            calls.append(parsed_call)
                        else:
                            calls.append({"name": match, "parameters": {}})
                    else:
                        calls.append({"name": match, "parameters": {}})
                except (json.JSONDecodeError, AttributeError):
                    calls.append({"name": match, "parameters": {}})

        return {
            "normal_text": normal_text,
            "calls": calls,
        }

    cleaned = re.sub(r"<\|im_end\|>", "", response).strip()
    parsed_call = _try_parse_json_tool_call(cleaned)
    if parsed_call:
        return {
            "normal_text": "",
            "calls": [parsed_call],
        }

    json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', re.DOTALL)
    json_matches = json_pattern.findall(response)
    if json_matches:
        calls = []
        for jm in json_matches:
            parsed_call = _try_parse_json_tool_call(jm)
            if parsed_call:
                calls.append(parsed_call)
        if calls:
            normal_text = json_pattern.sub("", response).strip()
            normal_text = re.sub(r"<\|im_end\|>", "", normal_text).strip()
            return {
                "normal_text": normal_text,
                "calls": calls,
            }

    return {
        "normal_text": response,
        "calls": [],
    }
