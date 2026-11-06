from dataclasses import dataclass


@dataclass
class Message:
    """One chat message replayed on the next model call."""

    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list | None = None


@dataclass
class Context:
    """One LiteLLM call: ordered messages with unified roles, and the tool catalog."""

    messages: list[Message]
    tools: list
