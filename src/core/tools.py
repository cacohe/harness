import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from src.core.artifacts import (
    ArtifactRef,
    ExternalizedContent,
    InlineContent,
    _decode_json,
    _encode_json,
)
from src.core.commands import _check_child_id
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import RunId, SessionId

_MAX_ARGUMENT_BYTES = 16_384
_MAX_SCHEMA_BYTES = 65_536
_MAX_DESCRIPTION_BYTES = 16_384
_WORKSPACE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,255}")


def _text_size(text: str, max_bytes: int) -> int:
    if type(text) is not str or len(text) > max_bytes:
        raise ValueError("Text must be an exact string within its byte limit")
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        raise ValueError("Text exceeds its byte limit")
    return size


def _canonical_json(text: str, max_bytes: int, *, allow_boolean_schema: bool = False) -> str:
    try:
        _text_size(text, max_bytes)
        value = _decode_json(text)
        if type(value) is not dict and not (allow_boolean_schema and type(value) is bool):
            raise ValueError
        return _encode_json(value, max_bytes)
    except ValueError:
        raise ValueError("Expected bounded strict JSON for tool arguments or schema") from None


def _canonical_arguments(text: str) -> str:
    return _canonical_json(text, _MAX_ARGUMENT_BYTES)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Tool catalog entry with a canonical JSON Schema snapshot."""

    name: str
    description: str
    input_schema_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_child_id(self.name)
        _text_size(self.description, _MAX_DESCRIPTION_BYTES)
        object.__setattr__(
            self,
            "input_schema_json",
            _canonical_json(self.input_schema_json, _MAX_SCHEMA_BYTES, allow_boolean_schema=True),
        )


class ToolApproval(StrEnum):
    UNDECIDED = "undecided"
    NOT_REQUIRED = "not_required"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """An invocation scoped to a run, step, workspace identity, and approval decision."""

    tool_call_id: str
    name: str
    arguments_json: str = field(repr=False)
    run_id: RunId
    session_id: SessionId
    step_id: str
    approval_scope: str
    workspace_id: str
    approval: ToolApproval

    def __post_init__(self) -> None:
        for identity in (self.tool_call_id, self.name, self.step_id, self.approval_scope):
            _check_child_id(identity)
        if type(self.run_id) is not RunId or type(self.session_id) is not SessionId:
            raise ValueError("Tool request requires typed run and session IDs")
        if (
            type(self.workspace_id) is not str
            or len(self.workspace_id) > 256
            or not _WORKSPACE_ID.fullmatch(self.workspace_id)
        ):
            raise ValueError("Tool request requires an opaque workspace ID")
        if type(self.approval) is not ToolApproval:
            raise ValueError("Tool request requires an explicit approval decision")
        object.__setattr__(self, "run_id", RunId(self.run_id.value))
        object.__setattr__(self, "session_id", SessionId(self.session_id.value))
        object.__setattr__(self, "arguments_json", _canonical_arguments(self.arguments_json))


class ToolResultStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    SUCCESS = "success"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


_FAILURE_CODES = {
    ToolResultStatus.FAILED: frozenset({RunErrorCode.TOOL_FAILED, RunErrorCode.ARTIFACT_FAILED}),
    ToolResultStatus.TIMED_OUT: frozenset({RunErrorCode.TOOL_TIMED_OUT}),
    ToolResultStatus.CANCELLED: frozenset({RunErrorCode.CANCELLED}),
}


@dataclass(frozen=True, slots=True, init=False)
class ToolResult:
    """One successful content variant, a stable failure, or a request for approval."""

    status: ToolResultStatus
    _content: ExternalizedContent | None = field(repr=False)
    _failure_code: RunErrorCode | None

    def __init__(
        self,
        status: ToolResultStatus,
        *,
        inline_content: InlineContent | None = None,
        artifacts: tuple[ArtifactRef, ...] = (),
        summary: str | None = None,
        failure: RunFailure | None = None,
    ) -> None:
        if type(status) is not ToolResultStatus:
            raise ValueError("Tool result requires a stable status")
        if type(artifacts) is not tuple:
            raise ValueError("Tool result artifacts must be a tuple")

        content = None
        failure_code = None
        if status is ToolResultStatus.SUCCESS:
            if failure is not None:
                raise ValueError("Successful tool results cannot carry a failure")
            content = ExternalizedContent(inline_content, artifacts, summary)
        else:
            if inline_content is not None or artifacts or summary is not None:
                raise ValueError("Unsuccessful tool results cannot carry execution content")
            if status is ToolResultStatus.PENDING_APPROVAL:
                if failure is not None:
                    raise ValueError("Pending approval cannot carry a failure")
            elif (
                type(failure) is not RunFailure
                or type(failure.code) is not RunErrorCode
                or failure.code not in _FAILURE_CODES[status]
            ):
                raise ValueError("Tool result requires a failure matching its status")
            else:
                failure_code = failure.code

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "_content", content)
        object.__setattr__(self, "_failure_code", failure_code)

    @property
    def inline_content(self) -> InlineContent | None:
        return self._content.inline_content if self._content is not None else None

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        return self._content.artifacts if self._content is not None else ()

    @property
    def summary(self) -> str | None:
        return self._content.summary if self._content is not None else None

    @property
    def failure(self) -> RunFailure | None:
        return RunFailure(self._failure_code) if self._failure_code is not None else None


class ToolGateway(Protocol):
    def execute(self, request: ToolRequest) -> Awaitable[ToolResult]:
        """Return a pending result for required approval or a settled execution result."""
        ...
