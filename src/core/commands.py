import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias
from unicodedata import category

from src.core.errors import InvalidTransition, RunFailure
from src.core.events import JsonValue, _contains_host_path, _encode_payload
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId

_MAX_CHILD_ID_BYTES = 256
_MAX_TOOL_ARGUMENT_BYTES = 16_384


@dataclass(frozen=True)
class RunLimits:
    max_steps: int
    max_elapsed_seconds: float
    max_inline_result_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_inline_result_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        elapsed = self.max_elapsed_seconds
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise ValueError("max_elapsed_seconds must be a positive finite number")
        try:
            finite = math.isfinite(elapsed)
        except OverflowError:
            finite = False
        if not finite or elapsed <= 0:
            raise ValueError("max_elapsed_seconds must be a positive finite number")


@dataclass(frozen=True)
class CommandContext:
    """Explicit transition time and a pool of event IDs in allocation order."""

    timestamp: datetime
    elapsed_seconds: float
    event_ids: tuple[EventId, ...] = ()

    def __post_init__(self) -> None:
        elapsed = self.elapsed_seconds
        if isinstance(elapsed, bool) or type(elapsed) not in (int, float) or elapsed < 0:
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        try:
            finite = math.isfinite(elapsed)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        if type(self.event_ids) is not tuple or not all(
            isinstance(event_id, EventId) for event_id in self.event_ids
        ):
            raise ValueError("event_ids must be a tuple of EventId values")


@dataclass(frozen=True)
class _RunCommand:
    run_id: RunId
    session_id: SessionId
    context: CommandContext = field(kw_only=True)


@dataclass(frozen=True)
class SubmitRun(_RunCommand):
    submission_id: SubmissionId
    profile_id: ProfileId
    user_input: str


@dataclass(frozen=True)
class ResumeRun(_RunCommand):
    pass


@dataclass(frozen=True)
class CancelRun(_RunCommand):
    pass


@dataclass(frozen=True)
class DecideToolCall(_RunCommand):
    tool_call_id: str
    approved: bool


@dataclass(frozen=True)
class StartStep(_RunCommand):
    step_id: str


@dataclass(frozen=True)
class FinishStep(_RunCommand):
    step_id: str


@dataclass(frozen=True)
class StartModelCall(_RunCommand):
    step_id: str
    model_call_id: str


@dataclass(frozen=True)
class ToolCallRequest:
    """Provider-independent request with immutable JSON argument text."""

    tool_call_id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _check_child_id(self.tool_call_id)
        _check_child_id(self.name)
        _check_bounded_text(self.arguments_json, _MAX_TOOL_ARGUMENT_BYTES)
        try:
            arguments = json.loads(self.arguments_json, object_pairs_hook=_unique_json_object)
            # Reuse the domain JSON rules after decoding escapes and rejecting duplicate keys.
            _encode_payload(arguments)
        except (ValueError, RecursionError):
            raise InvalidTransition from None


@dataclass(frozen=True)
class FinishModelCall(_RunCommand):
    step_id: str
    model_call_id: str
    result: str
    tool_request: ToolCallRequest | None = None


@dataclass(frozen=True)
class StartToolCall(_RunCommand):
    step_id: str
    model_call_id: str
    tool_call_id: str
    requires_approval: bool


@dataclass(frozen=True)
class FinishToolCall(_RunCommand):
    step_id: str
    model_call_id: str
    tool_call_id: str
    result: str | RunFailure


@dataclass(frozen=True)
class CompleteRun(_RunCommand):
    final_text: str


@dataclass(frozen=True)
class FailRun(_RunCommand):
    failure: RunFailure


CoreCommand: TypeAlias = (
    SubmitRun
    | ResumeRun
    | CancelRun
    | DecideToolCall
    | StartStep
    | FinishStep
    | StartModelCall
    | FinishModelCall
    | StartToolCall
    | FinishToolCall
    | CompleteRun
    | FailRun
)


def _check_child_id(value: str) -> None:
    _check_bounded_text(value, _MAX_CHILD_ID_BYTES)
    stripped = value.strip()
    if (
        not stripped
        or stripped in (".", "..")
        or "/" in stripped
        or "\\" in stripped
        or _contains_host_path(stripped)
        or any(category(character) == "Cc" for character in value)
    ):
        raise InvalidTransition
    if len(stripped) > 1 and stripped[0].isascii() and stripped[0].isalpha() and stripped[1] == ":":
        raise InvalidTransition


def _check_bounded_text(text: object, limit: int) -> None:
    if type(text) is not str or len(text) > limit:
        raise InvalidTransition
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise InvalidTransition from None
    if size > limit:
        raise InvalidTransition


def _unique_json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate tool argument key")
    return result
