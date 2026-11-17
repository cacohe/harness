from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm

from src.core.model import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from src.core.tools import ToolSpec

Completion = Callable[..., Awaitable[object]]


class ModelProviderError(RuntimeError):
    pass


@dataclass(slots=True)
class _CallParts:
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""


class LiteLLMModelGateway:
    """Wire-neutral LiteLLM adapter using only its native async API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout_seconds: float = 60.0,
        stream: bool = False,
        max_response_bytes: int = 1_048_576,
        max_tool_calls: int = 64,
        completion: Completion | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        if type(model) is not str or not model.strip() or len(model.encode()) > 512:
            raise ValueError("LiteLLM model is invalid")
        if api_key is not None and (type(api_key) is not str or len(api_key.encode()) > 16_384):
            raise ValueError("LiteLLM credential is invalid")
        if api_base is not None and (type(api_base) is not str or len(api_base.encode()) > 4096):
            raise ValueError("LiteLLM endpoint is invalid")
        if isinstance(timeout_seconds, bool) or type(timeout_seconds) not in (int, float):
            raise ValueError("LiteLLM timeout is invalid")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("LiteLLM timeout is invalid")
        if type(stream) is not bool:
            raise ValueError("LiteLLM stream mode is invalid")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 8_388_608:
            raise ValueError("LiteLLM response limit is invalid")
        if type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 64:
            raise ValueError("LiteLLM tool-call limit is invalid")
        if options is not None and type(options) is not dict:
            raise ValueError("LiteLLM options are invalid")
        forbidden = {"model", "messages", "tools", "stream", "api_key", "api_base", "timeout"}
        if options is not None and forbidden.intersection(options):
            raise ValueError("LiteLLM options override adapter-owned fields")
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout = timeout
        self._stream = stream
        self._max_bytes = max_response_bytes
        self._max_calls = max_tool_calls
        self._completion = completion or litellm.acompletion
        self._options = dict(options or {})

    def __repr__(self) -> str:
        return f"LiteLLMModelGateway(model={self._model!r}, stream={self._stream!r})"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        arguments: dict[str, object] = {
            **self._options,
            "model": self._model,
            "messages": [_message(item) for item in request.messages],
            "stream": self._stream,
            "timeout": self._timeout,
        }
        if request.tools:
            arguments["tools"] = [_tool(item) for item in request.tools]
        if self._api_key is not None:
            arguments["api_key"] = self._api_key
        if self._api_base is not None:
            arguments["api_base"] = self._api_base
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._completion(**arguments)
                if self._stream:
                    if not hasattr(response, "__aiter__"):
                        raise ValueError
                    return await self._assemble(response)  # type: ignore[arg-type]
                return self._parse(response)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            raise TimeoutError("Model request timed out") from None
        except Exception:
            raise ModelProviderError("Model request failed") from None

    def _parse(self, response: object) -> ModelResponse:
        choices = _field(response, "choices")
        if type(choices) is not list or len(choices) != 1:
            raise ValueError
        message = _field(choices[0], "message")
        text = _optional_text(_field(message, "content", None))
        raw_calls = _field(message, "tool_calls", [])
        if type(raw_calls) is not list or len(raw_calls) > self._max_calls:
            raise ValueError
        calls = tuple(_whole_call(item) for item in raw_calls)
        self._check_size(text, calls)
        return ModelResponse(text, calls, _usage(_field(response, "usage", None)))

    async def _assemble(self, response: AsyncIterator[object]) -> ModelResponse:
        text_parts: list[str] = []
        calls: list[_CallParts] = []
        usage: ModelUsage | None = None
        size = 0
        finished = False
        last_tool_index = -1
        async for chunk in response:
            chunk_usage = _usage(_field(chunk, "usage", None))
            if chunk_usage is not None:
                usage = chunk_usage
            choices = _field(chunk, "choices", [])
            if type(choices) is not list or len(choices) > 1:
                raise ValueError
            if not choices:
                continue
            choice = choices[0]
            finish = _field(choice, "finish_reason", None)
            if finish is not None:
                if type(finish) is not str or finished:
                    raise ValueError
                finished = True
            delta = _field(choice, "delta", None)
            if delta is None:
                continue
            content = _field(delta, "content", None)
            if content is not None:
                if type(content) is not str or finished and finish is None:
                    raise ValueError
                size += len(content.encode("utf-8"))
                if size > self._max_bytes:
                    raise ValueError
                text_parts.append(content)
            fragments = _field(delta, "tool_calls", [])
            if type(fragments) is not list:
                raise ValueError
            for fragment in fragments:
                index = _field(fragment, "index", None)
                if (
                    type(index) is not int
                    or index < 0
                    or index > len(calls)
                    or index < last_tool_index
                ):
                    raise ValueError
                last_tool_index = index
                if index == len(calls):
                    if len(calls) >= self._max_calls:
                        raise ValueError
                    calls.append(_CallParts())
                current = calls[index]
                call_id = _field(fragment, "id", None)
                function = _field(fragment, "function", {})
                name = _field(function, "name", None)
                arguments = _field(function, "arguments", None)
                if call_id is not None:
                    if type(call_id) is not str or current.call_id is not None:
                        raise ValueError
                    current.call_id = call_id
                if name is not None:
                    if type(name) is not str or current.name is not None:
                        raise ValueError
                    current.name = name
                if arguments is not None:
                    if type(arguments) is not str:
                        raise ValueError
                    size += len(arguments.encode("utf-8"))
                    if size > self._max_bytes:
                        raise ValueError
                    current.arguments += arguments
        model_calls = tuple(
            ModelToolCall(item.call_id or "", item.name or "", item.arguments) for item in calls
        )
        text = "".join(text_parts) or None
        self._check_size(text, model_calls)
        return ModelResponse(text, model_calls, usage)

    def _check_size(self, text: str | None, calls: tuple[ModelToolCall, ...]) -> None:
        size = len((text or "").encode("utf-8")) + sum(
            len(item.arguments_json.encode("utf-8")) for item in calls
        )
        if size > self._max_bytes:
            raise ValueError


def _message(message: ModelMessage) -> dict[str, object]:
    wire: dict[str, object] = {"role": message.role.value, "content": message.content}
    if message.role is ModelRole.ASSISTANT and message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.tool_call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in message.tool_calls
        ]
    if message.role is ModelRole.TOOL:
        wire["tool_call_id"] = message.tool_call_id
    return wire


def _tool(spec: ToolSpec) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": json.loads(spec.input_schema_json),
        },
    }


def _whole_call(value: object) -> ModelToolCall:
    function = _field(value, "function")
    call_id = _field(value, "id")
    name = _field(function, "name")
    arguments = _field(function, "arguments")
    if any(type(item) is not str for item in (call_id, name, arguments)):
        raise ValueError
    return ModelToolCall(call_id, name, arguments)


def _usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    input_tokens = _field(value, "prompt_tokens", None)
    output_tokens = _field(value, "completion_tokens", None)
    if type(input_tokens) is not int or type(output_tokens) is not int:
        raise ValueError
    return ModelUsage(input_tokens, output_tokens)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError
    return value


def _field(value: object, name: str, default: Any = ...) -> Any:
    if type(value) is dict:
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)
