import json
import re
from typing import Any


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = "qwen25") -> dict[str, Any]:
    if parser == "qwen25":
        return _parse_qwen25_tools(response)
    return _parse_qwen25_tools(response)


def _parse_qwen25_tools(response: str) -> dict[str, Any]:
    tool_call_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    matches = tool_call_pattern.findall(response)

    parts = tool_call_pattern.split(response)
    normal_text = parts[0].strip() if parts else ""

    calls = []
    for match in matches:
        match = match.strip()
        try:
            parsed = json.loads(match)
            name = parsed.get("name", "")
            parameters = parsed.get("arguments", parsed.get("parameters", {}))
            if isinstance(parameters, str):
                parameters = json.loads(parameters)
            calls.append({"name": name, "parameters": parameters})
        except json.JSONDecodeError:
            try:
                json_match = re.search(r"\{.*\}", match, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    name = parsed.get("name", "")
                    parameters = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(parameters, str):
                        parameters = json.loads(parameters)
                    calls.append({"name": name, "parameters": parameters})
            except (json.JSONDecodeError, AttributeError):
                calls.append({"name": match, "parameters": {}})

    return {
        "normal_text": normal_text,
        "calls": calls,
    }
