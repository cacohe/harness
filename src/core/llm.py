from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.core.context import Context

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


class LlmCallError(Exception):
    """A model call failed and will not be retried."""


class LlmGateway(ABC):
    """LLM Gateway"""

    @abstractmethod
    def planning(
        self,
        context: Context,
        stream: bool = True,
        response_format: Any | None = None,
    ) -> LLMOutput | Iterator[LLMOutput]:
        """
        Planning the task.
        Args:
            context: The context for the planning.
            stream: Whether to stream the response.
            response_format: Pydantic model or JSON schema dict the assistant text must follow.
        Returns:
            The planning output. When response_format is set, structured holds the parsed value.
        """
        ...

    @abstractmethod
    def infer(
        self,
        context: Context,
        stream: bool = True,
        response_format: Any | None = None,
    ) -> LLMOutput | Iterator[LLMOutput]:
        """
        Run one LLM call.
        Args:
            context: The context for the LLM call.
            stream: Whether to stream the response.
            response_format: Pydantic model or JSON schema dict the assistant text must follow.
        Returns:
            The LLM output. When response_format is set, structured holds the parsed value.
        """
        ...
