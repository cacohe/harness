from __future__ import annotations

import math
from collections.abc import Callable, Iterator

from src.core.context import EstimateQuality, TokenEstimate
from src.core.model import ModelMessage
from src.core.tools import ToolSpec

_MAX_TOKEN_COUNT = (1 << 63) - 1
_MAX_SAFETY_FACTOR = 1024.0


def _wire_text(messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]) -> Iterator[str]:
    for message in messages:
        yield message.role.value
        yield message.content or ""
        yield message.tool_call_id or ""
        for call in message.tool_calls:
            yield call.tool_call_id
            yield call.name
            yield call.arguments_json
    for tool in tools:
        yield tool.name
        yield tool.description
        yield tool.input_schema_json


class ExactTokenCounter:
    """An exact counter for a caller-supplied tokenizer.

    The default one-code-point tokenizer is intended for deterministic tests. A
    provider adapter should inject the tokenizer matching its wire format.
    """

    def __init__(self, count_text: Callable[[str], int] = len) -> None:
        if not callable(count_text):
            raise ValueError("Exact token counter requires a callable tokenizer")
        self._count_text = count_text

    def count(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> TokenEstimate:
        count = 0
        for text in _wire_text(messages, tools):
            value = self._count_text(text)
            if type(value) is not int or value < 0:
                raise ValueError("Exact tokenizer returned an invalid token count")
            count += value
        return TokenEstimate(count, EstimateQuality.EXACT)


class ConservativeTokenCounter:
    """Provider-neutral UTF-8 estimate with an explicit safety multiplier."""

    def __init__(self, *, chars_per_token: int = 3, safety_factor: float = 1.25) -> None:
        if (
            type(chars_per_token) is not int
            or chars_per_token <= 0
            or type(safety_factor) is not float
            or not math.isfinite(safety_factor)
            or not 1.0 <= safety_factor <= _MAX_SAFETY_FACTOR
        ):
            raise ValueError("Conservative counter configuration is invalid")
        self._chars_per_token = chars_per_token
        self._safety_factor = safety_factor

    def count(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> TokenEstimate:
        size = sum(len(value.encode("utf-8")) for value in _wire_text(messages, tools))
        envelope = 4 * len(messages) + 8 * len(tools)
        try:
            count = math.ceil((size / self._chars_per_token + envelope) * self._safety_factor)
        except OverflowError:
            raise ValueError("Conservative token estimate exceeds its supported range") from None
        if count > _MAX_TOKEN_COUNT:
            raise ValueError("Conservative token estimate exceeds its supported range")
        return TokenEstimate(count, EstimateQuality.CONSERVATIVE)
