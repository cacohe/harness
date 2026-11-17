from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from src.core.commands import _check_child_id
from src.core.tools import ToolSpec, _canonical_arguments, _text_size

_MAX_TEXT_BYTES = 1_048_576
_MAX_REQUEST_BYTES = 8 * 1_048_576
_MAX_MESSAGES = 1_024
_MAX_TOOLS = 128
_MAX_CALLS_PER_TURN = 64
_MAX_TOKEN_COUNT = (1 << 63) - 1


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    tool_call_id: str
    name: str
    arguments_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_child_id(self.tool_call_id)
        _check_child_id(self.name)
        object.__setattr__(self, "arguments_json", _canonical_arguments(self.arguments_json))


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """Text for system/user, text or calls for assistant, and text plus call ID for tool.

    An explicit empty string is a valid tool result. Assistant text and calls can coexist.
    """

    role: ModelRole
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.role) is not ModelRole:
            raise ValueError("Message requires a model role")
        if self.content is not None:
            _text_size(self.content, _MAX_TEXT_BYTES)

        if self.role is ModelRole.ASSISTANT:
            if self.tool_call_id is not None:
                raise ValueError("Assistant messages cannot reference a tool result")
            calls = _snapshot_calls(self.tool_calls)
            _require_assistant_output(self.content, calls)
            object.__setattr__(self, "tool_calls", calls)
            return

        if type(self.tool_calls) is not tuple or self.tool_calls or self.content is None:
            raise ValueError("System, user, and tool messages require text and no tool calls")
        if self.role is ModelRole.TOOL:
            if self.tool_call_id is None:
                raise ValueError("A tool message requires a call ID")
            _check_child_id(self.tool_call_id)
        elif self.tool_call_id is not None or not self.content.strip():
            raise ValueError("System and user messages require non-blank text and no call ID")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A message window with each tool batch contiguous and paired by call ID.

    Results may arrive in any order within their batch. Only a trailing batch may
    be missing results.
    """

    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolSpec, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.messages) is not tuple
            or not 1 <= len(self.messages) <= _MAX_MESSAGES
            or any(type(message) is not ModelMessage for message in self.messages)
        ):
            raise ValueError("Model request requires a bounded, nonempty tuple of messages")
        if (
            type(self.tools) is not tuple
            or len(self.tools) > _MAX_TOOLS
            or any(type(tool) is not ToolSpec for tool in self.tools)
        ):
            raise ValueError("Model request requires a bounded tuple of tool specs")

        seen_calls: set[str] = set()
        pending: set[str] = set()
        messages: list[ModelMessage] = []
        size = 0
        for original in self.messages:
            message = replace(original)
            if message.role is ModelRole.TOOL:
                call_id = message.tool_call_id
                if call_id is None or call_id not in pending:
                    raise ValueError("Tool result requires a pending call in its assistant turn")
                pending.remove(call_id)
            elif pending:
                raise ValueError("Tool results must precede the next message turn")
            else:
                pending = {call.tool_call_id for call in message.tool_calls}
                if not seen_calls.isdisjoint(pending):
                    raise ValueError("Tool call IDs must be unique within a model request")
                seen_calls.update(pending)
            size += _message_size(message)
            if size > _MAX_REQUEST_BYTES:
                raise ValueError("Model request exceeds its byte limit")
            messages.append(message)

        tools: list[ToolSpec] = []
        tool_names: set[str] = set()
        for original_tool in self.tools:
            tool = replace(original_tool)
            if tool.name in tool_names:
                raise ValueError("Tool names must be unique within a model request")
            tool_names.add(tool.name)
            size += sum(
                len(value.encode("utf-8"))
                for value in (tool.name, tool.description, tool.input_schema_json)
            )
            if size > _MAX_REQUEST_BYTES:
                raise ValueError("Model request exceeds its byte limit")
            tools.append(tool)

        object.__setattr__(self, "messages", tuple(messages))
        object.__setattr__(self, "tools", tuple(tools))


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if any(
            type(count) is not int or not 0 <= count <= _MAX_TOKEN_COUNT
            for count in (self.input_tokens, self.output_tokens)
        ):
            raise ValueError("Token usage requires exact nonnegative integer counts")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Assistant text, requested tool calls, or both, with optional token counts."""

    text: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        if self.text is not None:
            _text_size(self.text, _MAX_TEXT_BYTES)
        calls = _snapshot_calls(self.tool_calls)
        _require_assistant_output(self.text, calls)
        object.__setattr__(self, "tool_calls", calls)
        if self.usage is not None:
            if type(self.usage) is not ModelUsage:
                raise ValueError("Response usage must be a model token count")
            object.__setattr__(self, "usage", replace(self.usage))


def _snapshot_calls(calls: tuple[ModelToolCall, ...]) -> tuple[ModelToolCall, ...]:
    if (
        type(calls) is not tuple
        or len(calls) > _MAX_CALLS_PER_TURN
        or any(type(call) is not ModelToolCall for call in calls)
    ):
        raise ValueError("Assistant tool calls must be a bounded tuple of model tool calls")
    snapshot = tuple(replace(call) for call in calls)
    if len({call.tool_call_id for call in snapshot}) != len(snapshot):
        raise ValueError("Tool call IDs must be unique within an assistant turn")
    return snapshot


def _require_assistant_output(text: str | None, calls: tuple[ModelToolCall, ...]) -> None:
    if not calls and (text is None or not text.strip()):
        raise ValueError("Assistant output requires non-blank text or a tool call")


def _message_size(message: ModelMessage) -> int:
    size = sum(
        len(value.encode("utf-8"))
        for value in (message.role.value, message.content or "", message.tool_call_id or "")
    )
    for call in message.tool_calls:
        size += sum(
            len(value.encode("utf-8"))
            for value in (call.tool_call_id, call.name, call.arguments_json)
        )
    return size


class ModelGateway(Protocol):
    def complete(self, request: ModelRequest) -> Awaitable[ModelResponse]: ...
