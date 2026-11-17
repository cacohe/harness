from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from src.core.artifacts import ArtifactRef, ExternalizedContent, InlineContent
from src.core.commands import ToolCallRequest, _check_child_id
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import RunId, SessionId
from src.core.json_values import validate_bounded_text, validate_json_text

_MAX_SCHEMA_BYTES = 65_536
_MAX_DESCRIPTION_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Tool catalog entry with an immutable JSON Schema snapshot."""

    name: str
    description: str
    input_schema_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_child_id(self.name)
        validate_bounded_text(self.description, _MAX_DESCRIPTION_BYTES)
        validate_json_text(self.input_schema_json, _MAX_SCHEMA_BYTES, allow_boolean_root=True)


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
    original_call: ToolCallRequest | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        _check_child_id(self.step_id)
        if type(self.run_id) is not RunId or type(self.session_id) is not SessionId:
            raise ValueError("Tool request requires typed run and session IDs")
        if type(self.approval) is not ToolApproval:
            raise ValueError("Tool request requires an explicit approval decision")

        call = ToolCallRequest(
            self.tool_call_id,
            self.name,
            self.arguments_json,
            approval_scope=self.approval_scope,
            workspace_id=self.workspace_id,
        )
        original = self.original_call
        if original is not None:
            if type(original) is not ToolCallRequest:
                raise ValueError("Tool request requires a valid original call")
            original = replace(original)
            if original != call:
                raise ValueError("Tool request must match its original call and approval binding")
            object.__setattr__(self, "original_call", original)
        elif self.approval in (ToolApproval.APPROVED, ToolApproval.DENIED):
            raise ValueError("A decided approval requires the original call")

        object.__setattr__(self, "run_id", RunId(self.run_id.value))
        object.__setattr__(self, "session_id", SessionId(self.session_id.value))

    @classmethod
    def from_call(
        cls,
        call: ToolCallRequest,
        *,
        run_id: RunId,
        session_id: SessionId,
        step_id: str,
        approval: ToolApproval,
    ) -> "ToolRequest":
        """Carry the persisted call and its original approval binding into execution."""
        if type(call) is not ToolCallRequest:
            raise ValueError("Tool request requires a valid original call")
        return cls(
            tool_call_id=call.tool_call_id,
            name=call.name,
            arguments_json=call.arguments_json,
            run_id=run_id,
            session_id=session_id,
            step_id=step_id,
            approval_scope=call.approval_scope,
            workspace_id=call.workspace_id,
            approval=approval,
            original_call=call,
        )

    @property
    def is_executable(self) -> bool:
        """Whether the decision permits execution, subject to live adapter validation."""
        return self.approval in (ToolApproval.NOT_REQUIRED, ToolApproval.APPROVED)

    @property
    def requires_approval(self) -> bool:
        """Whether a decision is still required before execution."""
        return self.approval is ToolApproval.UNDECIDED


class ToolResultStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    DENIED = "denied"
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
            inline = content.inline_content
            if inline is not None and inline.media_type != "application/json":
                raise ValueError("Inline tool results require application/json")
        else:
            if inline_content is not None or artifacts or summary is not None:
                raise ValueError("Unsuccessful tool results cannot carry execution content")
            if status in (ToolResultStatus.PENDING_APPROVAL, ToolResultStatus.DENIED):
                if failure is not None:
                    raise ValueError("A pending or denied approval cannot carry a failure")
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
        """Revalidate the current approval policy, scope, and workspace before any side effect.

        Return PENDING_APPROVAL for UNDECIDED and DENIED for a denied or stale
        approval without invoking the tool. Dispatch only executable requests
        whose original binding still matches the active tool and workspace.
        """
        ...
