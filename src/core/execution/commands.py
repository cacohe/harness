from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias
from unicodedata import category

from src.core.artifacts.contracts import ArtifactRef, ExternalizedContent, snapshot_artifact_ref
from src.core.execution.errors import InvalidTransition, RunFailure
from src.core.execution.events import _contains_host_path
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.json_values import MAX_TOOL_ARGUMENT_BYTES, validate_json_text
from src.core.tools.approval import validate_approval_binding

if TYPE_CHECKING:
    from src.core.context._skill_values import SkillBinding

_MAX_CHILD_ID_BYTES = 256


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
        if len({event_id.value for event_id in self.event_ids}) != len(self.event_ids):
            raise InvalidTransition


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
class AdvanceRun(_RunCommand):
    """Request one coordinator-owned model/tool step for a running Run."""


@dataclass(frozen=True)
class ActivateSkill(_RunCommand):
    binding: SkillBinding

    def __post_init__(self) -> None:
        from src.core.context._skill_values import snapshot_skill_binding

        try:
            object.__setattr__(self, "binding", snapshot_skill_binding(self.binding))
        except ValueError:
            raise InvalidTransition from None


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
    """Persistent call identity, approval binding, and lossless JSON arguments."""

    tool_call_id: str
    name: str
    arguments_json: str = field(repr=False)
    requires_approval: bool = field(kw_only=True)
    approval_scope: str = field(kw_only=True)
    workspace_id: str = field(kw_only=True)
    available: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        _check_child_id(self.tool_call_id)
        _check_child_id(self.name)
        if type(self.requires_approval) is not bool:
            raise InvalidTransition
        if type(self.available) is not bool or (not self.available and self.requires_approval):
            raise InvalidTransition
        try:
            validate_approval_binding(self.approval_scope, self.workspace_id)
            validate_json_text(self.arguments_json, MAX_TOOL_ARGUMENT_BYTES)
        except ValueError:
            raise InvalidTransition from None


def snapshot_tool_call(request: object) -> ToolCallRequest:
    """Rebuild from declared fields without reading instance dataclass metadata."""
    if type(request) is not ToolCallRequest:
        raise InvalidTransition
    try:
        return ToolCallRequest(
            object.__getattribute__(request, "tool_call_id"),
            object.__getattribute__(request, "name"),
            object.__getattribute__(request, "arguments_json"),
            requires_approval=object.__getattribute__(request, "requires_approval"),
            approval_scope=object.__getattribute__(request, "approval_scope"),
            workspace_id=object.__getattribute__(request, "workspace_id"),
            available=object.__getattribute__(request, "available"),
        )
    except AttributeError:
        raise InvalidTransition from None


@dataclass(frozen=True)
class FinishModelCall(_RunCommand):
    step_id: str
    model_call_id: str
    result: str
    tool_requests: tuple[ToolCallRequest, ...] = ()

    def __post_init__(self) -> None:
        if type(self.tool_requests) is not tuple:
            raise InvalidTransition
        requests = self.tool_requests
        if len(requests) > 64:
            raise InvalidTransition
        snapshots = tuple(snapshot_tool_call(request) for request in requests)
        if len({request.tool_call_id for request in snapshots}) != len(snapshots):
            raise InvalidTransition
        object.__setattr__(self, "tool_requests", snapshots)


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
    result: ExternalizedContent | RunFailure


@dataclass(frozen=True)
class CompleteRun(_RunCommand):
    final_text: str
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.artifacts) is not tuple
            or len(self.artifacts) > 16
            or any(type(ref) is not ArtifactRef for ref in self.artifacts)
        ):
            raise InvalidTransition
        try:
            snapshots = tuple(snapshot_artifact_ref(ref) for ref in self.artifacts)
        except ValueError:
            raise InvalidTransition from None
        object.__setattr__(self, "artifacts", snapshots)


@dataclass(frozen=True)
class FailRun(_RunCommand):
    failure: RunFailure


CoreCommand: TypeAlias = (
    SubmitRun
    | AdvanceRun
    | ActivateSkill
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
