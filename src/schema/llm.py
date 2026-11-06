from dataclasses import dataclass
from enum import Enum
from typing import Any

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


class FinishReason(str, Enum):
    """Why the model stopped generating."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"

    def requests_tools(self) -> bool:
        """
        Report whether the model stopped to call a tool.
        Returns:
            True for tool_calls and function_call.
        """
        return self in {FinishReason.TOOL_CALLS, FinishReason.FUNCTION_CALL}


@dataclass
class LlmToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class LlmUsage:
    """Token counts for one model call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMOutput:
    """Model output used by the agent loop."""

    finish_reason: FinishReason | None
    content: str | None
    tool_calls: list[LlmToolCall]
    reasoning_content: str | None
    thinking_blocks: list[dict[str, Any]] | None
    usage: LlmUsage | None = None
    structured: Any | None = None

    def requests_tool_calls(self) -> bool:
        """
        Report whether this output asks the harness to run tool calls.
        Returns:
            True when the finish reason asks for tools and at least one call is present.
        """
        if self.finish_reason is None or not self.tool_calls:
            return False
        return self.finish_reason.requests_tools()
