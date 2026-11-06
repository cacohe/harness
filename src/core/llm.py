from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from src.schema.context import Context
from src.schema.llm import FinishReason, LLMOutput


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


def get_finish_reason_value(finish_reason: FinishReason | None) -> str | None:
    """
    Get the value of the finish reason.
    Args:
        finish_reason: The finish reason to get the value of.
    Returns:
        The value of the finish reason.
    """
    if finish_reason is None:
        return None
    return finish_reason.value
