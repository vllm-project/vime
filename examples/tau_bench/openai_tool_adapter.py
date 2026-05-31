import json
import logging
from dataclasses import dataclass
from typing import Any

from .sglang_tool_parser import parse_tools
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
from tau_bench.types import Action

logger = logging.getLogger(__name__)


@dataclass
class OpenAIToolCall:
    id: str
    type: str = "function"
    function: dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    def __init__(self, tools_info: list[dict[str, Any]], parser_type: str = "qwen25"):
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> dict[str, Any]:
        try:
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]
            openai_message = self._convert_to_openai_message(normal_text, calls)
            return {"openai_message": openai_message, "parsed_result": parsed, "success": True}
        except Exception as e:
            logger.warning(f"Parsing failed with error: {str(e)}")
            return {"openai_message": None, "parsed_result": None, "success": False, "error": str(e)}

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        if not calls:
            return OpenAIAssistantMessage(role="assistant", content=normal_text, tool_calls=None)

        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={"name": call.get("name", ""), "arguments": call.get("parameters", "{}")},
            )
            openai_tool_calls.append(openai_tool_call)

        return OpenAIAssistantMessage(
            role="assistant", content=normal_text if normal_text.strip() else None, tool_calls=openai_tool_calls
        )

    def _call_to_action_sglang(self, calls: list[Any], text_response: str) -> Action:
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
        if calls:
            if len(calls) > 1:
                logger.debug("Multiple tool calls identified, only taking first.")
            tool_call = calls[0]
            try:
                params = json.loads(tool_call["parameters"])
                if not isinstance(params, dict):
                    logger.warning(f"{params} does not follow dict structure for action")
                else:
                    action = Action(name=tool_call["name"], kwargs=params)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters as JSON: {e}")
        return action

    def get_openai_tools_format(self) -> list[dict[str, Any]]:
        openai_tools = []
        for tool in self.tools_info:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools


def create_openai_adapter(
    tools_info: list[dict[str, Any]], parser_type: str = "qwen25"
) -> OpenAICompatibleToolCallAdapter:
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
