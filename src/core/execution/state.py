from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypeAlias, assert_never

from src.core.artifacts.contracts import ArtifactRef, ExternalizedContent, snapshot_artifact_ref
from src.core.execution.commands import (
    AdvanceRun,
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
    snapshot_tool_call,
)
from src.core.execution.errors import InvalidTransition, RunErrorCode, RunFailure
from src.core.execution.events import DomainEvent, DomainEventType, JsonValue
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId


class RunStatus(StrEnum):
    """Lifecycle states persisted in the current Run projection."""

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
    """Two-phase approval: a decision is recorded before the Run is resumed."""

    approved: bool | None = None
    resumed: bool = False


@dataclass(frozen=True)
class ToolCallState:
    """A persisted tool request together with approval and execution progress."""

    request: ToolCallRequest
    requires_approval: bool
    approval: ApprovalState | None = None
    result: ExternalizedContent | RunFailure | None = None

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
    """The model result and ordered tool batch produced for one step."""

    model_call_id: str
    result: str | None = None
    tool_requests: tuple[ToolCallRequest, ...] = ()

    @property
    def tool_request(self) -> ToolCallRequest | None:
        return self.tool_requests[0] if len(self.tool_requests) == 1 else None

    @property
    def tool_call_id(self) -> str | None:
        return self.tool_request.tool_call_id if self.tool_request else None


@dataclass(frozen=True)
class StepState:
    """One model decision plus its serially processed tool calls."""

    step_id: str
    model_call: ModelCallState | None = None
    tool_calls: tuple[ToolCallState, ...] = ()
    # Durable cursor pointing at the next tool request that must be settled.
    next_tool_index: int = 0

    @property
    def tool_call(self) -> ToolCallState | None:
        if not self.tool_calls:
            return None
        index = min(self.next_tool_index, len(self.tool_calls) - 1)
        return self.tool_calls[index]


@dataclass(frozen=True)
class RunState:
    """Immutable durable projection of a Run and its current in-flight step."""

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
    final_artifacts: tuple[ArtifactRef, ...] = ()
    failure: RunFailure | None = None
    journal_version: int = 0
    journal_sequence: int = 0

    def __post_init__(self) -> None:
        from src.core.execution.state_validation import validate_state

        validate_state(self)

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
    """Validate commands and derive the next immutable state plus domain event."""

    @staticmethod
    def apply(state: RunState, command: CoreCommand) -> StateTransition:
        if command.run_id != state.run_id or command.session_id != state.session_id:
            raise InvalidTransition
        elapsed = command.context.elapsed_seconds
        if state.status.terminal or elapsed < state.elapsed_seconds:
            raise InvalidTransition

        advancing = replace(state, version=state.version + 1)
        next_state, fact = _advance(advancing, command)
        # Time exhaustion wins over ordinary progress, but explicit close commands remain stable.
        if (
            not isinstance(command, (CancelRun, FailRun))
            and elapsed > state.limits.max_elapsed_seconds
        ):
            next_state, fact = _fail(advancing, RunFailure(RunErrorCode.LIMIT_EXCEEDED))

        event_ids = command.context.event_ids
        events: tuple[DomainEvent, ...] = ()
        if fact:
            # Each emitted fact consumes the matching prefix of caller-allocated event IDs.
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
    # CREATED and WAITING_APPROVAL have narrow command sets; all other work is RUNNING-only.
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
    if isinstance(command, (SubmitRun, AdvanceRun, DecideToolCall, ResumeRun)):
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
            return _close(
                state,
                RunStatus.COMPLETED,
                final_text=command.final_text,
                final_artifacts=command.artifacts,
            ), (
                DomainEventType.RUN_COMPLETED,
                {"artifact_ids": [ref.artifact_id for ref in command.artifacts]},
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
            if step.next_tool_index != len(model.tool_requests) or any(
                not tool.settled for tool in step.tool_calls
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
            requests = tuple(snapshot_tool_call(request) for request in command.tool_requests)
            model = replace(model, result=command.result, tool_requests=requests)
            return replace(state, current_step=replace(step, model_call=model)), None
        case StartToolCall():
            return _start_tool(state, command)
        case FinishToolCall():
            step = _step(state, command.step_id)
            _model(step, command.model_call_id)
            _check_child_id(command.tool_call_id)
            if not 0 <= step.next_tool_index < len(step.tool_calls):
                raise InvalidTransition
            tool = step.tool_calls[step.next_tool_index]
            if tool is None or tool.tool_call_id != command.tool_call_id or tool.settled:
                raise InvalidTransition
            if type(command.result) is ExternalizedContent:
                try:
                    result: ExternalizedContent | RunFailure = ExternalizedContent(
                        command.result.inline_content,
                        command.result.artifacts,
                        command.result.summary,
                    )
                except ValueError:
                    raise InvalidTransition from None
            elif type(command.result) is RunFailure:
                try:
                    result = RunFailure(command.result.code, command.result.detail)
                except ValueError:
                    raise InvalidTransition from None
            else:
                raise InvalidTransition
            tool = replace(tool, result=result)
            tools = (
                *step.tool_calls[: step.next_tool_index],
                tool,
                *step.tool_calls[step.next_tool_index + 1 :],
            )
            return replace(
                state,
                current_step=replace(
                    step, tool_calls=tools, next_tool_index=step.next_tool_index + 1
                ),
            ), None
    assert_never(command)


def _start_tool(state: RunState, command: StartToolCall) -> tuple[RunState, _EventFact | None]:
    step = _step(state, command.step_id)
    model = _model(step, command.model_call_id)
    _check_child_id(command.tool_call_id)
    if not 0 <= step.next_tool_index < len(model.tool_requests):
        raise InvalidTransition
    request = model.tool_requests[step.next_tool_index]
    if (
        request is None
        or request.tool_call_id != command.tool_call_id
        or len(step.tool_calls) != step.next_tool_index
    ):
        raise InvalidTransition
    if (
        type(command.requires_approval) is not bool
        or command.requires_approval is not request.requires_approval
    ):
        raise InvalidTransition
    approval = ApprovalState() if command.requires_approval else None
    tool = ToolCallState(request, requires_approval=command.requires_approval, approval=approval)
    next_state = replace(
        state,
        current_step=replace(step, tool_calls=(*step.tool_calls, tool)),
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
    if step is None or not step.tool_calls or step.next_tool_index >= len(step.tool_calls):
        raise InvalidTransition
    tool = step.tool_calls[step.next_tool_index]
    approval = tool.approval
    if approval is None:
        raise InvalidTransition

    if isinstance(command, ResumeRun) and approval.approved is not None:
        # A rejected call is settled by advancing the cursor; an approved call executes next.
        tool = replace(tool, approval=replace(approval, resumed=True))
        next_index = step.next_tool_index + (approval.approved is False)
        tools = (
            *step.tool_calls[: step.next_tool_index],
            tool,
            *step.tool_calls[step.next_tool_index + 1 :],
        )
        return replace(
            state,
            status=RunStatus.RUNNING,
            current_step=replace(step, tool_calls=tools, next_tool_index=next_index),
        ), (DomainEventType.RUN_RESUMED, {})
    if isinstance(command, DecideToolCall):
        _check_child_id(command.tool_call_id)
    if (
        isinstance(command, DecideToolCall)
        and command.tool_call_id == tool.tool_call_id
        and approval.approved is None
        and type(command.approved) is bool
    ):
        # Recording the decision deliberately keeps the Run paused until an explicit resume.
        tool = replace(tool, approval=ApprovalState(command.approved))
        tools = (
            *step.tool_calls[: step.next_tool_index],
            tool,
            *step.tool_calls[step.next_tool_index + 1 :],
        )
        return replace(state, current_step=replace(step, tool_calls=tools)), (
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


def _close(
    state: RunState,
    status: RunStatus,
    *,
    final_text: str | None = None,
    final_artifacts: tuple[ArtifactRef, ...] = (),
    failure: RunFailure | None = None,
) -> RunState:
    try:
        artifacts = tuple(snapshot_artifact_ref(ref) for ref in final_artifacts)
    except ValueError:
        raise InvalidTransition from None
    return replace(
        state,
        status=status,
        current_step=None,
        final_text=final_text,
        final_artifacts=artifacts,
        failure=failure,
    )


def _fail(state: RunState, failure: RunFailure) -> tuple[RunState, _EventFact]:
    return _close(state, RunStatus.FAILED, failure=failure), (
        DomainEventType.RUN_FAILED,
        {"code": failure.code.value},
    )
