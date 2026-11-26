import math

from src.core.artifacts.contracts import ArtifactRef, ExternalizedContent
from src.core.context.skills import snapshot_skill_bindings
from src.core.execution.commands import (
    RunLimits,
    ToolCallRequest,
    _check_bounded_text,
    _check_child_id,
)
from src.core.execution.errors import InvalidTransition, RunErrorCode, RunFailure
from src.core.execution.state import (
    ApprovalState,
    ModelCallState,
    RunState,
    RunStatus,
    StepState,
    ToolCallState,
)
from src.core.ids import ProfileId, RunId, SessionId, SubmissionId


def validate_state(state: RunState) -> None:
    try:
        active_skills = snapshot_skill_bindings(state.active_skills)
    except ValueError:
        raise InvalidTransition from None
    if active_skills != state.active_skills:
        raise InvalidTransition
    if (
        type(state.run_id) is not RunId
        or type(state.session_id) is not SessionId
        or type(state.submission_id) is not SubmissionId
        or type(state.profile_id) is not ProfileId
        or type(state.limits) is not RunLimits
        or type(state.status) is not RunStatus
        or type(state.version) is not int
        or state.version < 0
        or type(state.journal_version) is not int
        or state.journal_version < 0
        or type(state.journal_sequence) is not int
        or state.journal_sequence < 0
        or type(state.used_step_ids) is not frozenset
        or type(state.used_tool_call_ids) is not frozenset
        or state.steps_started > state.limits.max_steps
        or len(state.used_tool_call_ids) > state.limits.max_steps * 64
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
        _bounded(state, state.final_text)
        if (
            type(state.final_artifacts) is not tuple
            or len(state.final_artifacts) > 16
            or any(type(ref) is not ArtifactRef for ref in state.final_artifacts)
        ):
            raise InvalidTransition
    elif state.final_text is not None or state.final_artifacts:
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
    requests: tuple[ToolCallRequest, ...] = ()
    model = step.model_call
    if model is not None:
        if type(model) is not ModelCallState:
            raise InvalidTransition
        _check_child_id(model.model_call_id)
        if model.result is not None:
            _bounded(state, model.result)
        if type(model.tool_requests) is not tuple or len(model.tool_requests) > 64:
            raise InvalidTransition
        requests = model.tool_requests
        if requests:
            if model.result is None:
                raise InvalidTransition
            seen: set[str] = set()
            for request in requests:
                if type(request) is not ToolCallRequest or request.tool_call_id in seen:
                    raise InvalidTransition
                seen.add(request.tool_call_id)
    if (
        type(step.tool_calls) is not tuple
        or type(step.next_tool_index) is not int
        or not 0 <= step.next_tool_index <= len(step.tool_calls)
    ):
        raise InvalidTransition
    tools = step.tool_calls
    waiting = state.status is RunStatus.WAITING_APPROVAL
    if not tools:
        if (
            waiting
            or step.next_tool_index != 0
            or any(request.tool_call_id in state.used_tool_call_ids for request in requests)
        ):
            raise InvalidTransition
        return
    if len(tools) > len(requests) or len(tools) not in (
        step.next_tool_index,
        step.next_tool_index + 1,
    ):
        raise InvalidTransition
    for index, tool in enumerate(tools):
        _validate_tool(state, tool, requests[index], index < step.next_tool_index)
    active = tools[step.next_tool_index] if step.next_tool_index < len(tools) else None
    if waiting != (
        active is not None and active.approval is not None and not active.approval.resumed
    ):
        raise InvalidTransition


def _validate_tool(
    state: RunState, tool: ToolCallState, request: ToolCallRequest, before_cursor: bool
) -> None:
    if (
        type(tool) is not ToolCallState
        or type(tool.request) is not ToolCallRequest
        or tool.request != request
        or tool.request.tool_call_id not in state.used_tool_call_ids
        or type(tool.requires_approval) is not bool
    ):
        raise InvalidTransition
    approval = tool.approval
    if tool.requires_approval != (approval is not None):
        raise InvalidTransition
    if approval is not None and (
        type(approval) is not ApprovalState
        or type(approval.resumed) is not bool
        or (approval.approved is not None and type(approval.approved) is not bool)
        or (approval.resumed and approval.approved is None)
    ):
        raise InvalidTransition
    if before_cursor and not tool.settled:
        raise InvalidTransition
    if tool.result is not None and type(tool.result) not in (ExternalizedContent, RunFailure):
        raise InvalidTransition
    if type(tool.result) is ExternalizedContent:
        inline = tool.result.inline_content
        if inline is not None and inline.size_bytes > state.limits.max_inline_result_bytes:
            raise InvalidTransition
        if tool.result.summary is not None:
            _bounded(state, tool.result.summary)
    if (
        tool.result is not None
        and approval is not None
        and (approval.approved is False or not approval.resumed)
    ):
        raise InvalidTransition


def _bounded(state: RunState, text: object) -> None:
    _check_bounded_text(text, state.limits.max_inline_result_bytes)
