import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypeAlias, assert_never

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
    ToolCallRequest,
    _check_bounded_text,
    _check_child_id,
)
from src.core.errors import InvalidTransition, RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType, JsonValue
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId


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
    resumed: bool = False


@dataclass(frozen=True)
class ToolCallState:
    request: ToolCallRequest
    approval: ApprovalState | None = None
    result: str | RunFailure | None = None

    @property
    def tool_call_id(self) -> str:
        return self.request.tool_call_id

    @property
    def settled(self) -> bool:
        return self.result is not None or (
            self.approval is not None and self.approval.approved is False
        )


@dataclass(frozen=True)
class ModelCallState:
    model_call_id: str
    result: str | None = None
    tool_request: ToolCallRequest | None = None

    @property
    def tool_call_id(self) -> str | None:
        return self.tool_request.tool_call_id if self.tool_request else None


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

    def __post_init__(self) -> None:
        _validate_state(self)

    @classmethod
    def create(cls, command: SubmitRun, limits: RunLimits, *, version: int = 0) -> "RunState":
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
    """Events consume the corresponding prefix of the supplied event IDs."""

    state: RunState
    events: tuple[DomainEvent, ...]
    remaining_event_ids: tuple[EventId, ...]


_EventFact: TypeAlias = tuple[DomainEventType, dict[str, JsonValue]]


class TaskStateMachine:
    @staticmethod
    def apply(state: RunState, command: CoreCommand) -> StateTransition:
        if command.run_id != state.run_id or command.session_id != state.session_id:
            raise InvalidTransition
        elapsed = command.context.elapsed_seconds
        if state.status.terminal or elapsed < state.elapsed_seconds:
            raise InvalidTransition

        advancing = replace(state, version=state.version + 1)
        next_state, fact = _advance(advancing, command)
        if (
            not isinstance(command, (CancelRun, FailRun))
            and elapsed > state.limits.max_elapsed_seconds
        ):
            next_state, fact = _fail(advancing, RunFailure(RunErrorCode.LIMIT_EXCEEDED))

        event_ids = command.context.event_ids
        events: tuple[DomainEvent, ...] = ()
        if fact:
            if not event_ids:
                raise InvalidTransition
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
            replace(next_state, elapsed_seconds=elapsed)
            if elapsed != next_state.elapsed_seconds
            else next_state,
            events,
            event_ids[len(events) :],
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
            _check_child_id(command.step_id)
            if state.current_step is not None or command.step_id in state.used_step_ids:
                raise InvalidTransition
            if state.steps_started >= state.limits.max_steps:
                return _fail(state, RunFailure(RunErrorCode.LIMIT_EXCEEDED))
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
            model = replace(model, result=command.result, tool_request=command.tool_request)
            return replace(state, current_step=replace(step, model_call=model)), None
        case StartToolCall():
            return _start_tool(state, command)
        case FinishToolCall():
            step = _step(state, command.step_id)
            _model(step, command.model_call_id)
            _check_child_id(command.tool_call_id)
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
    _check_child_id(command.tool_call_id)
    request = model.tool_request
    if (
        request is None
        or request.tool_call_id != command.tool_call_id
        or step.tool_call is not None
    ):
        raise InvalidTransition
    if type(command.requires_approval) is not bool:
        raise InvalidTransition
    approval = ApprovalState() if command.requires_approval else None
    tool = ToolCallState(request, approval=approval)
    next_state = replace(
        state,
        current_step=replace(step, tool_call=tool),
        used_tool_call_ids=state.used_tool_call_ids | {request.tool_call_id},
        status=RunStatus.RUNNING if approval is None else RunStatus.WAITING_APPROVAL,
    )
    if approval is None:
        return next_state, None
    return next_state, (
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
        tool = replace(tool, approval=replace(approval, resumed=True))
        return replace(
            state, status=RunStatus.RUNNING, current_step=replace(step, tool_call=tool)
        ), (DomainEventType.RUN_RESUMED, {})
    if isinstance(command, DecideToolCall):
        _check_child_id(command.tool_call_id)
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
    _check_child_id(step_id)
    step = state.current_step
    if step is None or step.step_id != step_id:
        raise InvalidTransition
    return step


def _model(step: StepState, model_call_id: str) -> ModelCallState:
    _check_child_id(model_call_id)
    model = step.model_call
    if model is None or model.model_call_id != model_call_id:
        raise InvalidTransition
    return model


def _check_text(state: RunState, text: object) -> None:
    _check_bounded_text(text, state.limits.max_inline_result_bytes)


def _validate_state(state: RunState) -> None:
    if (
        type(state.run_id) is not RunId
        or type(state.session_id) is not SessionId
        or type(state.submission_id) is not SubmissionId
        or type(state.profile_id) is not ProfileId
        or type(state.limits) is not RunLimits
        or type(state.status) is not RunStatus
        or type(state.version) is not int
        or state.version < 0
        or type(state.used_step_ids) is not frozenset
        or type(state.used_tool_call_ids) is not frozenset
        or not len(state.used_tool_call_ids) <= state.steps_started <= state.limits.max_steps
    ):
        raise InvalidTransition
    for identifiers in (state.used_step_ids, state.used_tool_call_ids):
        for identifier in identifiers:
            _check_child_id(identifier)

    elapsed = state.elapsed_seconds
    if type(elapsed) not in (int, float) or elapsed < 0:
        raise InvalidTransition
    try:
        if not math.isfinite(elapsed):
            raise InvalidTransition
    except OverflowError:
        raise InvalidTransition from None

    status = state.status
    if status is RunStatus.CREATED:
        if elapsed != 0 or state.used_step_ids:
            raise InvalidTransition
    elif state.version <= state.steps_started:
        raise InvalidTransition
    if status in (RunStatus.FAILED, RunStatus.CANCELLED):
        if type(state.failure) is not RunFailure or (
            (state.failure.code is RunErrorCode.CANCELLED) != (status is RunStatus.CANCELLED)
        ):
            raise InvalidTransition
    elif state.failure is not None or elapsed > state.limits.max_elapsed_seconds:
        raise InvalidTransition
    if status is RunStatus.COMPLETED:
        _check_text(state, state.final_text)
    elif state.final_text is not None:
        raise InvalidTransition

    if state.current_step is None:
        if status is RunStatus.WAITING_APPROVAL:
            raise InvalidTransition
    elif status in (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL):
        _validate_step(state, state.current_step)
    else:
        raise InvalidTransition


def _validate_step(state: RunState, step: StepState) -> None:
    if type(step) is not StepState:
        raise InvalidTransition
    _check_child_id(step.step_id)
    if step.step_id not in state.used_step_ids:
        raise InvalidTransition

    request = None
    model = step.model_call
    if model is not None:
        if type(model) is not ModelCallState:
            raise InvalidTransition
        _check_child_id(model.model_call_id)
        if model.result is not None:
            _check_text(state, model.result)
        request = model.tool_request
        if request is not None:
            if type(request) is not ToolCallRequest or model.result is None:
                raise InvalidTransition

    tool = step.tool_call
    waiting = state.status is RunStatus.WAITING_APPROVAL
    if tool is None:
        if (
            waiting
            or len(state.used_tool_call_ids) >= state.steps_started
            or (request is not None and request.tool_call_id in state.used_tool_call_ids)
        ):
            raise InvalidTransition
        return
    if (
        type(tool) is not ToolCallState
        or request is None
        or type(tool.request) is not ToolCallRequest
        or tool.request != request
        or request.tool_call_id not in state.used_tool_call_ids
    ):
        raise InvalidTransition

    approval = tool.approval
    if approval is not None:
        if (
            type(approval) is not ApprovalState
            or type(approval.resumed) is not bool
            or (approval.approved is not None and type(approval.approved) is not bool)
            or (approval.resumed and approval.approved is None)
        ):
            raise InvalidTransition
    if waiting != (approval is not None and not approval.resumed):
        raise InvalidTransition
    if tool.result is not None:
        if waiting or (approval is not None and approval.approved is False):
            raise InvalidTransition
        if type(tool.result) is not RunFailure:
            _check_text(state, tool.result)


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
