from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypeAlias, assert_never
from unicodedata import category

from src.core.commands import (
    CancelRun,
    CompleteRun,
    CoreCommand,
    DecideToolCall,
    FailRun,
    FinishModelCall,
    FinishStep,
    FinishToolCall,
    ResumeRun,
    RunLimits,
    StartModelCall,
    StartStep,
    StartToolCall,
    SubmitRun,
)
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType, JsonValue
from src.core.ids import ProfileId, RunId, SessionId, SubmissionId


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


@dataclass(frozen=True)
class ApprovalState:
    approved: bool | None = None


@dataclass(frozen=True)
class ToolCallState:
    tool_call_id: str
    approval: ApprovalState | None = None
    result: str | RunFailure | None = None

    @property
    def settled(self) -> bool:
        return self.result is not None or (
            self.approval is not None and self.approval.approved is False
        )


@dataclass(frozen=True)
class ModelCallState:
    model_call_id: str
    result: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class StepState:
    step_id: str
    model_call: ModelCallState | None = None
    tool_call: ToolCallState | None = None


@dataclass(frozen=True)
class RunState:
    run_id: RunId
    session_id: SessionId
    submission_id: SubmissionId
    profile_id: ProfileId
    limits: RunLimits
    version: int
    status: RunStatus = RunStatus.CREATED
    current_step: StepState | None = None
    elapsed_seconds: float = 0
    used_step_ids: frozenset[str] = frozenset()
    used_tool_call_ids: frozenset[str] = frozenset()
    final_text: str | None = None
    failure: RunFailure | None = None

    @classmethod
    def create(cls, command: SubmitRun, limits: RunLimits, *, version: int = 0) -> "RunState":
        if type(version) is not int or version < 0:
            raise ValueError("version must be a non-negative integer")
        return cls(
            run_id=command.run_id,
            session_id=command.session_id,
            submission_id=command.submission_id,
            profile_id=command.profile_id,
            limits=limits,
            version=version,
        )

    @property
    def steps_started(self) -> int:
        return len(self.used_step_ids)


@dataclass(frozen=True)
class StateTransition:
    state: RunState
    events: tuple[DomainEvent, ...]


class InvalidTransition(ValueError):
    def __init__(self) -> None:
        self.failure = RunFailure(RunErrorCode.INVARIANT_VIOLATION)
        super().__init__(self.failure.message)


_EventFact: TypeAlias = tuple[DomainEventType, dict[str, JsonValue]]


class TaskStateMachine:
    @staticmethod
    def apply(state: RunState, command: CoreCommand) -> StateTransition:
        if command.run_id != state.run_id or command.session_id != state.session_id:
            raise InvalidTransition
        elapsed = command.context.elapsed_seconds
        if state.status.terminal or elapsed < state.elapsed_seconds:
            raise InvalidTransition

        next_state, fact = _advance(state, command)
        if not isinstance(command, (CancelRun, FailRun)) and (
            elapsed > state.limits.max_elapsed_seconds
            or next_state.steps_started > state.limits.max_steps
        ):
            next_state, fact = _fail(state, RunFailure(RunErrorCode.LIMIT_EXCEEDED))

        event_ids = command.context.event_ids
        if len(event_ids) != (1 if fact else 0):
            raise InvalidTransition
        events: tuple[DomainEvent, ...] = ()
        if fact:
            event_type, payload = fact
            events = (
                DomainEvent(
                    event_ids[0],
                    event_type,
                    state.run_id,
                    state.session_id,
                    command.context.timestamp,
                    payload,
                ),
            )
        return StateTransition(
            replace(next_state, version=state.version + 1, elapsed_seconds=elapsed),
            events,
        )


def _advance(state: RunState, command: CoreCommand) -> tuple[RunState, _EventFact | None]:
    if state.status is RunStatus.CREATED:
        if not isinstance(command, SubmitRun) or (
            command.submission_id != state.submission_id or command.profile_id != state.profile_id
        ):
            raise InvalidTransition
        return replace(state, status=RunStatus.RUNNING), (DomainEventType.RUN_STARTED, {})

    if isinstance(command, CancelRun):
        failure = RunFailure(RunErrorCode.CANCELLED)
        return _close(state, RunStatus.CANCELLED, failure=failure), (
            DomainEventType.RUN_CANCELLED,
            {"code": failure.code.value},
        )
    if isinstance(command, FailRun):
        return _fail(state, command.failure)
    if state.status is RunStatus.WAITING_APPROVAL:
        return _approval(state, command)
    if isinstance(command, (SubmitRun, DecideToolCall, ResumeRun)):
        raise InvalidTransition
    return _advance_running(state, command)


def _advance_running(
    state: RunState,
    command: (
        StartStep
        | FinishStep
        | StartModelCall
        | FinishModelCall
        | StartToolCall
        | FinishToolCall
        | CompleteRun
    ),
) -> tuple[RunState, _EventFact | None]:
    match command:
        case CompleteRun():
            if state.current_step is not None:
                raise InvalidTransition
            _check_text(state, command.final_text)
            return _close(state, RunStatus.COMPLETED, final_text=command.final_text), (
                DomainEventType.RUN_COMPLETED,
                {},
            )
        case StartStep():
            if state.current_step is not None or command.step_id in state.used_step_ids:
                raise InvalidTransition
            _check_child_id(command.step_id)
            return replace(
                state,
                current_step=StepState(command.step_id),
                used_step_ids=state.used_step_ids | {command.step_id},
            ), None
        case FinishStep():
            step = _step(state, command.step_id)
            model = step.model_call
            if model is None or model.result is None:
                raise InvalidTransition
            if model.tool_call_id is not None and (
                step.tool_call is None or not step.tool_call.settled
            ):
                raise InvalidTransition
            return replace(state, current_step=None), None
        case StartModelCall():
            step = _step(state, command.step_id)
            if step.model_call is not None:
                raise InvalidTransition
            _check_child_id(command.model_call_id)
            step = replace(step, model_call=ModelCallState(command.model_call_id))
            return replace(state, current_step=step), None
        case FinishModelCall():
            step = _step(state, command.step_id)
            model = _model(step, command.model_call_id)
            if model.result is not None:
                raise InvalidTransition
            _check_text(state, command.result)
            if command.tool_call_id is not None:
                _check_child_id(command.tool_call_id)
                if command.tool_call_id in state.used_tool_call_ids:
                    raise InvalidTransition
            model = replace(model, result=command.result, tool_call_id=command.tool_call_id)
            return replace(state, current_step=replace(step, model_call=model)), None
        case StartToolCall():
            return _start_tool(state, command)
        case FinishToolCall():
            step = _step(state, command.step_id)
            _model(step, command.model_call_id)
            tool = step.tool_call
            if tool is None or tool.tool_call_id != command.tool_call_id or tool.settled:
                raise InvalidTransition
            if isinstance(command.result, str):
                _check_text(state, command.result)
            elif not isinstance(command.result, RunFailure):
                raise InvalidTransition
            tool = replace(tool, result=command.result)
            return replace(state, current_step=replace(step, tool_call=tool)), None
    assert_never(command)


def _start_tool(state: RunState, command: StartToolCall) -> tuple[RunState, _EventFact | None]:
    step = _step(state, command.step_id)
    model = _model(step, command.model_call_id)
    if (
        model.result is None
        or model.tool_call_id != command.tool_call_id
        or step.tool_call is not None
    ):
        raise InvalidTransition
    if type(command.requires_approval) is not bool:
        raise InvalidTransition
    approval = ApprovalState() if command.requires_approval else None
    tool = ToolCallState(command.tool_call_id, approval=approval)
    next_state = replace(
        state,
        current_step=replace(step, tool_call=tool),
        used_tool_call_ids=state.used_tool_call_ids | {command.tool_call_id},
    )
    if approval is None:
        return next_state, None
    return replace(next_state, status=RunStatus.WAITING_APPROVAL), (
        DomainEventType.RUN_WAITING_APPROVAL,
        {"tool_call_id": command.tool_call_id},
    )


def _approval(state: RunState, command: CoreCommand) -> tuple[RunState, _EventFact]:
    step = state.current_step
    if step is None or step.tool_call is None:
        raise InvalidTransition
    tool = step.tool_call
    approval = tool.approval
    if approval is None:
        raise InvalidTransition

    if isinstance(command, ResumeRun) and approval.approved is not None:
        return replace(state, status=RunStatus.RUNNING), (DomainEventType.RUN_RESUMED, {})
    if (
        isinstance(command, DecideToolCall)
        and command.tool_call_id == tool.tool_call_id
        and approval.approved is None
        and type(command.approved) is bool
    ):
        tool = replace(tool, approval=ApprovalState(command.approved))
        return replace(state, current_step=replace(step, tool_call=tool)), (
            DomainEventType.TOOL_CALL_DECIDED,
            {"tool_call_id": tool.tool_call_id, "approved": command.approved},
        )
    raise InvalidTransition


def _step(state: RunState, step_id: str) -> StepState:
    step = state.current_step
    if step is None or step.step_id != step_id:
        raise InvalidTransition
    return step


def _model(step: StepState, model_call_id: str) -> ModelCallState:
    model = step.model_call
    if model is None or model.model_call_id != model_call_id:
        raise InvalidTransition
    return model


def _check_child_id(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or len(value.encode("utf-8")) > 256
        or any(category(character) == "Cc" for character in value)
    ):
        raise InvalidTransition


def _check_text(state: RunState, text: str) -> None:
    limit = state.limits.max_inline_result_bytes
    if type(text) is not str or len(text) > limit or len(text.encode("utf-8")) > limit:
        raise InvalidTransition


def _close(
    state: RunState,
    status: RunStatus,
    *,
    final_text: str | None = None,
    failure: RunFailure | None = None,
) -> RunState:
    return replace(state, status=status, current_step=None, final_text=final_text, failure=failure)


def _fail(state: RunState, failure: RunFailure) -> tuple[RunState, _EventFact]:
    return _close(state, RunStatus.FAILED, failure=failure), (
        DomainEventType.RUN_FAILED,
        {"code": failure.code.value},
    )
