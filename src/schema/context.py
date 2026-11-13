from typing import Any

from pydantic import BaseModel, Field

from src.schema.llm import LlmToolCall


class Message(BaseModel):
    """One chat message replayed on the next model call."""

    role: str
    content: str | None = None
    tool_calls: list[LlmToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None


class Context(BaseModel):
    """One LiteLLM call: ordered messages with unified roles, and the tool catalog."""

    messages: list[dict[str, Any] | Message] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
