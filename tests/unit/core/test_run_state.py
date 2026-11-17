import asyncio
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from itertools import product

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.core.commands import (
    CancelRun,
    CommandContext,
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
)
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import JournalCommit
from src.core.state import ApprovalState, InvalidTransition, RunState, RunStatus, TaskStateMachine
from src.core.tools import ToolApproval, ToolRequest

RUN = RunId("run")
SESSION = SessionId("session")
NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)
FAILURE = RunFailure(RunErrorCode.MODEL_FAILED)
LIMITS = RunLimits(max_steps=2, max_elapsed_seconds=10, max_inline_result_bytes=32)


def context(event_id: str | None = None, *, elapsed: float = 0) -> CommandContext:
    return CommandContext(NOW, elapsed, (EventId(event_id),) if event_id else ())


def submit() -> SubmitRun:
    return SubmitRun(
        RUN,
        SESSION,
        SubmissionId("submission"),
        ProfileId("profile"),
        "hello",
        context=context("start"),
    )


def initial(limits: RunLimits = LIMITS) -> RunState:
    return RunState.create(submit(), limits, version=7)


def advance(state: RunState, *commands: CoreCommand) -> RunState:
    for command in commands:
        state = TaskStateMachine.apply(state, command).state
    return state


def begin_step(
    step_id: str = "step", tool_call_id: str | None = "tool"
) -> tuple[StartStep, StartModelCall, FinishModelCall]:
    request = (
        ToolCallRequest(tool_call_id, "lookup", "{}", approval_scope="read", workspace_id="work")
        if tool_call_id is not None
        else None
    )
    return (
        StartStep(RUN, SESSION, step_id, context=context()),
        StartModelCall(RUN, SESSION, step_id, "model", context=context()),
        FinishModelCall(RUN, SESSION, step_id, "model", "model result", request, context=context()),
    )


def ask_approval(step_id: str = "step", tool_call_id: str = "tool") -> StartToolCall:
    return StartToolCall(
        RUN,
        SESSION,
        step_id,
        "model",
        tool_call_id,
        requires_approval=True,
        context=context(f"ask-{step_id}"),
    )


def decide(approved: bool = True, tool_call_id: str = "tool") -> DecideToolCall:
    return DecideToolCall(RUN, SESSION, tool_call_id, approved, context=context("decision"))


def resume() -> ResumeRun:
    return ResumeRun(RUN, SESSION, context=context("resume"))


def finish_tool(step_id: str = "step", tool_call_id: str = "tool") -> FinishToolCall:
    return FinishToolCall(RUN, SESSION, step_id, "model", tool_call_id, "result", context=context())


def finish_step(step_id: str = "step") -> FinishStep:
    return FinishStep(RUN, SESSION, step_id, context=context())


def complete() -> CompleteRun:
    return CompleteRun(RUN, SESSION, "answer", context=context("complete"))


def test_approval_flow_emits_only_ordered_facts_and_keeps_previous_states():
    cases = (
        (submit(), RunStatus.RUNNING, DomainEventType.RUN_STARTED),
        *[(command, RunStatus.RUNNING, None) for command in begin_step()],
        (ask_approval(), RunStatus.WAITING_APPROVAL, DomainEventType.RUN_WAITING_APPROVAL),
        (decide(), RunStatus.WAITING_APPROVAL, DomainEventType.TOOL_CALL_DECIDED),
        (resume(), RunStatus.RUNNING, DomainEventType.RUN_RESUMED),
        (finish_tool(), RunStatus.RUNNING, None),
        (finish_step(), RunStatus.RUNNING, None),
        (complete(), RunStatus.COMPLETED, DomainEventType.RUN_COMPLETED),
    )
    state = created = initial()
    emitted_ids = []
    for command, status, event_type in cases:
        previous = state
        transition = TaskStateMachine.apply(state, command)
        assert transition == TaskStateMachine.apply(previous, command)
        state = transition.state
        assert state is not previous
        assert state.status is status
        assert state.version == previous.version + 1
        assert tuple(event.type for event in transition.events) == (
            (event_type,) if event_type else ()
        )
        for event in transition.events:
            emitted_ids.append(event.event_id.value)
            assert (event.run_id, event.session_id, event.timestamp) == (RUN, SESSION, NOW)

    assert emitted_ids == ["start", "ask-step", "decision", "resume", "complete"]
    assert state.final_text == "answer"
    assert state.current_step is None
    assert (created.status, created.version, created.current_step) == (RunStatus.CREATED, 7, None)
    with pytest.raises(FrozenInstanceError):
        setattr(state, "status", RunStatus.CREATED)


@pytest.mark.parametrize(
    ("terminal", "status", "event_type", "code"),
    [
        (
            CompleteRun(RUN, SESSION, "answer", context=context("complete")),
            RunStatus.COMPLETED,
            DomainEventType.RUN_COMPLETED,
            None,
        ),
        (
            FailRun(RUN, SESSION, FAILURE, context=context("fail")),
            RunStatus.FAILED,
            DomainEventType.RUN_FAILED,
            RunErrorCode.MODEL_FAILED,
        ),
        (
            CancelRun(RUN, SESSION, context=context("cancel")),
            RunStatus.CANCELLED,
            DomainEventType.RUN_CANCELLED,
            RunErrorCode.CANCELLED,
        ),
    ],
)
def test_terminal_run_cannot_process_more_commands(terminal, status, event_type, code):
    started = advance(initial(), submit())
    transition = TaskStateMachine.apply(started, terminal)
    state = transition.state
    assert state.status is status
    assert state.failure == (RunFailure(code) if code else None)
    assert [event.type for event in transition.events] == [event_type]
    for command in (submit(), resume(), decide(), terminal, *begin_step()):
        with pytest.raises(InvalidTransition):
            TaskStateMachine.apply(state, command)


@pytest.mark.parametrize(
    ("field", "wrong_id"),
    [("run_id", RunId("foreign")), ("session_id", SessionId("foreign"))],
)
def test_approval_is_owned_by_both_run_and_session(field, wrong_id):
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    with pytest.raises(InvalidTransition) as exc:
        TaskStateMachine.apply(waiting, replace(decide(), **{field: wrong_id}))
    assert exc.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert advance(waiting, decide(), resume()).status is RunStatus.RUNNING


def test_start_requires_the_matching_submission_and_profile():
    for command in (
        replace(submit(), submission_id=SubmissionId("other")),
        replace(submit(), profile_id=ProfileId("other")),
        resume(),
    ):
        with pytest.raises(InvalidTransition):
            TaskStateMachine.apply(initial(), command)


@pytest.mark.parametrize("approved", [True, False])
def test_pending_approval_is_decided_once_and_requires_resume(approved):
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    for command in (resume(), decide(tool_call_id="different"), finish_tool()):
        with pytest.raises(InvalidTransition):
            TaskStateMachine.apply(waiting, command)

    decision = TaskStateMachine.apply(waiting, decide(approved))
    assert decision.events[0].payload == {"tool_call_id": "tool", "approved": approved}
    with pytest.raises(InvalidTransition):
        TaskStateMachine.apply(decision.state, decide(not approved))
    resumed = advance(decision.state, resume())
    if approved:
        settled = advance(resumed, finish_tool())
    else:
        with pytest.raises(InvalidTransition):
            TaskStateMachine.apply(resumed, finish_tool())
        settled = resumed
    with pytest.raises(InvalidTransition):
        TaskStateMachine.apply(settled, finish_tool())
    assert advance(settled, finish_step(), complete()).status is RunStatus.COMPLETED


def test_cancel_waiting_closes_child_lifecycles_without_a_tool_result():
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    cancelled = advance(waiting, CancelRun(RUN, SESSION, context=context("cancel", elapsed=11)))
    assert cancelled.status is RunStatus.CANCELLED
    assert cancelled.current_step is None
    with pytest.raises(InvalidTransition):
        TaskStateMachine.apply(cancelled, decide())


def test_child_lifecycles_require_the_current_step_and_matching_parent_call():
    running = advance(initial(), submit())
    start_step, start_model, finish_model = begin_step()
    start_tool = replace(ask_approval(), requires_approval=False, context=context())
    step_open = advance(running, start_step)
    model_open = advance(step_open, start_model)
    model_settled = advance(model_open, finish_model)
    tool_open = advance(model_settled, start_tool)
    for state, command in (
        (running, start_model),
        (step_open, replace(start_step, step_id="second")),
        (step_open, replace(start_model, step_id="second")),
        (model_open, start_model),
        (model_open, replace(finish_model, model_call_id="other")),
        (model_open, start_tool),
        (model_open, finish_step()),
        (model_settled, finish_model),
        (model_settled, replace(start_tool, tool_call_id="unplanned")),
        (model_settled, replace(start_tool, model_call_id="other")),
        (model_settled, finish_step()),
        (tool_open, start_tool),
        (tool_open, replace(finish_tool(), step_id="other")),
        (tool_open, replace(finish_tool(), model_call_id="other")),
        (tool_open, replace(finish_tool(), tool_call_id="other")),
        (tool_open, finish_step()),
        (tool_open, complete()),
    ):
        with pytest.raises(InvalidTransition):
            TaskStateMachine.apply(state, command)

    settled = advance(tool_open, finish_tool())
    assert settled.current_step is not None
    assert settled.current_step.tool_call is not None
    assert settled.current_step.tool_call.result == "result"
    assert advance(settled, finish_step()).current_step is None


def test_settled_tool_id_cannot_be_scheduled_under_a_second_step():
    running = advance(
        initial(),
        submit(),
        *begin_step(),
        ask_approval(),
        decide(),
        resume(),
        finish_tool(),
        finish_step(),
    )
    with pytest.raises(InvalidTransition):
        advance(running, *begin_step("second", "tool"))


def test_step_budget_allows_the_last_step_but_fails_on_the_next_start():
    state = advance(
        initial(RunLimits(1, 10, 32)), submit(), *begin_step(tool_call_id=None), finish_step()
    )
    assert state.status is RunStatus.RUNNING
    assert state.steps_started == 1
    transition = TaskStateMachine.apply(
        state, StartStep(RUN, SESSION, "second", context=context("limit"))
    )
    assert transition.state.status is RunStatus.FAILED
    assert transition.state.failure == RunFailure(RunErrorCode.LIMIT_EXCEEDED)
    assert transition.state.steps_started == 1
    assert transition.events[0].type is DomainEventType.RUN_FAILED
    assert transition.events[0].payload["code"] == "limit_exceeded"


def test_elapsed_budget_uses_only_explicit_monotonic_time():
    running = advance(initial(), submit())
    at_limit = advance(running, StartStep(RUN, SESSION, "step", context=context(elapsed=10)))
    assert at_limit.elapsed_seconds == 10
    with pytest.raises(InvalidTransition):
        advance(at_limit, StartModelCall(RUN, SESSION, "step", "model", context=context(elapsed=9)))
    transition = TaskStateMachine.apply(
        at_limit,
        StartModelCall(RUN, SESSION, "step", "model", context=context("timeout", elapsed=10.001)),
    )
    assert transition.state.status is RunStatus.FAILED
    assert transition.state.failure == RunFailure(RunErrorCode.LIMIT_EXCEEDED)
    assert transition.state.current_step is None
    assert transition.events[0].event_id == EventId("timeout")


def test_inline_result_budget_measures_utf8_bytes():
    running = advance(initial(RunLimits(2, 10, 12)), submit(), *begin_step())
    started = advance(running, replace(ask_approval(), requires_approval=False, context=context()))
    at_limit = advance(started, replace(finish_tool(), result="é" * 6))
    assert at_limit.current_step is not None
    assert at_limit.current_step.tool_call is not None
    assert at_limit.current_step.tool_call.result == "é" * 6
    with pytest.raises(InvalidTransition):
        advance(started, replace(finish_tool(), result="é" * 7))


def test_event_metadata_must_cover_the_events_actually_emitted():
    with pytest.raises(InvalidTransition):
        advance(initial(), replace(submit(), context=context()))


def test_generated_sequences_preserve_state_machine_invariants():
    # Toggle an invalid command independently at nine lifecycle boundaries.
    # Both terminal outcomes are exercised with every combination of invalid insertions.
    first_step = begin_step()
    second_step = begin_step("second", "another-tool")
    second_tool = replace(
        ask_approval("second", "another-tool"), requires_approval=False, context=context()
    )
    legal_pairs = (
        (resume(), submit()),
        (first_step[1], first_step[0]),
        (first_step[2], first_step[1]),
        (ask_approval(), first_step[2]),
        (finish_tool(), ask_approval()),
        (resume(), decide()),
        (decide(False), resume()),
        (finish_step(), finish_tool()),
        (finish_tool(), finish_step()),
    )
    tail = (*second_step, second_tool, finish_tool("second", "another-tool"), finish_step("second"))
    for *toggles, cancel in product((False, True), repeat=len(legal_pairs) + 1):
        state = initial()
        settled_ids: set[str] = set()
        terminal = CancelRun(RUN, SESSION, context=context("cancel")) if cancel else complete()
        sequence = [
            (invalid if enabled else None, valid)
            for enabled, (invalid, valid) in zip(toggles, legal_pairs, strict=True)
        ] + [(None, command) for command in (*tail, terminal)]
        for invalid, valid in sequence:
            if invalid is not None:
                with pytest.raises(InvalidTransition):
                    TaskStateMachine.apply(state, invalid)
            previous = state
            state = advance(state, valid)
            assert state.version > previous.version
            step = state.current_step
            if step and step.tool_call and step.tool_call.settled:
                previous_tool = previous.current_step.tool_call if previous.current_step else None
                if previous_tool is None or not previous_tool.settled:
                    assert step.tool_call.tool_call_id not in settled_ids
                    settled_ids.add(step.tool_call.tool_call_id)
            if step and step.tool_call:
                assert step.model_call is not None
                assert step.model_call.tool_call_id == step.tool_call.tool_call_id
            assert len(state.used_tool_call_ids) <= state.steps_started <= LIMITS.max_steps
        assert settled_ids == {"tool", "another-tool"}
        assert state.steps_started == len(state.used_tool_call_ids) == LIMITS.max_steps
        assert state.status is (RunStatus.CANCELLED if cancel else RunStatus.COMPLETED)
        assert state.current_step is None
        for command in (submit(), resume(), decide(), finish_tool()):
            with pytest.raises(InvalidTransition):
                TaskStateMachine.apply(state, command)


def test_normal_and_limited_start_accept_same_event_pool():
    limits = RunLimits(1, 10, 32)
    ready = advance(initial(limits), submit())
    exhausted = advance(ready, *begin_step(tool_call_id=None), finish_step())
    pool = (EventId("first"), EventId("second"))
    command = StartStep(RUN, SESSION, "new", context=CommandContext(NOW, 0, pool))

    normal = TaskStateMachine.apply(ready, command)
    limited = TaskStateMachine.apply(exhausted, command)
    assert normal.events == ()
    assert normal.remaining_event_ids == pool
    assert tuple(event.event_id for event in limited.events) == pool[:1]
    assert limited.remaining_event_ids == pool[1:]
    assert limited.state.failure == RunFailure(RunErrorCode.LIMIT_EXCEEDED)
    with pytest.raises(InvalidTransition):
        TaskStateMachine.apply(exhausted, replace(command, context=context()))


def test_event_pool_rejects_equal_ids_before_they_can_be_consumed():
    duplicate = (EventId("shared"), EventId("fresh"), EventId("shared"))
    with pytest.raises(InvalidTransition) as error:
        CommandContext(NOW, 0, duplicate)
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": SessionId("run")},
        {"session_id": "session"},
        {"submission_id": ProfileId("submission")},
        {"profile_id": SubmissionId("profile")},
        {"version": True},
        {"version": -1},
        {"version": 0},
        {"elapsed_seconds": float("nan")},
        {"elapsed_seconds": 11},
        {"used_step_ids": frozenset({"a", "b", "c"})},
        {"used_step_ids": ["step", "step"]},
        {"used_tool_call_ids": frozenset({"a", "b", "c"})},
        {"used_tool_call_ids": frozenset({"unowned"})},
        {"final_text": "premature"},
        {"failure": FAILURE},
        {"status": "running"},
        {"status": RunStatus.WAITING_APPROVAL},
        {"status": RunStatus.COMPLETED},
    ],
)
def test_running_snapshot_rejects_invalid_direct_construction_and_replacement(changes):
    running = advance(initial(), submit())
    with pytest.raises(InvalidTransition):
        replace(running, **changes)
    with pytest.raises(InvalidTransition):
        RunState(**{**vars(running), **changes})


def test_snapshot_cannot_skip_resume_or_smuggle_result_while_waiting():
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    approved = advance(waiting, decide())
    for state in (waiting, approved):
        with pytest.raises(InvalidTransition):
            replace(state, status=RunStatus.RUNNING)
    assert approved.current_step is not None
    assert approved.current_step.tool_call is not None
    with pytest.raises(InvalidTransition):
        replace(
            approved,
            current_step=replace(
                approved.current_step,
                tool_call=replace(approved.current_step.tool_call, result="injected"),
            ),
        )
    with pytest.raises(InvalidTransition):
        replace(
            waiting,
            status=RunStatus.COMPLETED,
            final_text="answer",
        )


@pytest.mark.parametrize("outcome", ["pending", "approved", "resumed", "denied"])
def test_required_approval_cannot_be_erased_from_restored_tool(outcome):
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    match outcome:
        case "pending":
            snapshot = waiting
        case "approved":
            snapshot = advance(waiting, decide())
        case "resumed":
            snapshot = advance(waiting, decide(), resume())
        case _:
            snapshot = advance(waiting, decide(False), resume())
    step = snapshot.current_step
    assert step is not None and step.tool_call is not None
    with pytest.raises(InvalidTransition):
        replace(
            snapshot,
            status=RunStatus.RUNNING,
            current_step=replace(step, tool_call=replace(step.tool_call, approval=None)),
        )


def test_tool_without_required_approval_can_settle_but_cannot_gain_approval():
    command = replace(ask_approval(), requires_approval=False, context=context())
    running = advance(initial(), submit(), *begin_step(), command)
    step = running.current_step
    assert running.status is RunStatus.RUNNING
    assert step is not None and step.tool_call is not None
    assert step.tool_call.approval is None
    assert advance(RunState(**vars(running)), finish_tool(), finish_step()).current_step is None
    with pytest.raises(InvalidTransition):
        replace(
            running,
            current_step=replace(
                step,
                tool_call=replace(step.tool_call, approval=ApprovalState(True, resumed=True)),
            ),
        )


def test_snapshot_rejects_missing_or_mismatched_child_ownership():
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    step = waiting.current_step
    assert step is not None
    assert step.tool_call is not None
    for changes in (
        {"used_step_ids": frozenset({"other"})},
        {"used_tool_call_ids": frozenset({"other"})},
        {"current_step": replace(step, model_call=None)},
        {
            "current_step": replace(
                step,
                tool_call=replace(
                    step.tool_call, request=replace(step.tool_call.request, tool_call_id="other")
                ),
            )
        },
    ):
        with pytest.raises(InvalidTransition):
            replace(waiting, **changes)


@pytest.mark.parametrize(
    "value",
    ["/tool123", r"C:\tool123", "file:tool", "use file:secret", "\ud800", "a\nb", "é" * 129],
)
@pytest.mark.parametrize("kind", ["step", "model", "tool"])
def test_rejects_unsafe_ids_at_first_acceptance(kind, value):
    running = advance(initial(), submit())
    start_step, start_model, finish_model = begin_step()
    match kind:
        case "step":
            state, command = running, replace(start_step, step_id=value)
        case "model":
            state = advance(running, start_step)
            command = replace(start_model, model_call_id=value)
        case _:
            state = advance(running, start_step, start_model)
            command = finish_model
    with pytest.raises(InvalidTransition) as error:
        if kind == "tool":
            command = replace(
                finish_model,
                tool_request=ToolCallRequest(
                    value, "lookup", "{}", approval_scope="read", workspace_id="work"
                ),
            )
        TaskStateMachine.apply(state, command)
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_approved_tool_request_can_be_restored_and_resumed():
    request = ToolCallRequest(
        "tool", "read_file", '{"path":"README.md"}', approval_scope="read", workspace_id="work"
    )
    start_step, start_model, finish_model = begin_step()
    model_done = advance(
        initial(), submit(), start_step, start_model, replace(finish_model, tool_request=request)
    )
    step = model_done.current_step
    assert step is not None and step.model_call is not None
    assert step.model_call.tool_request == request
    transition = TaskStateMachine.apply(RunState(**vars(model_done)), ask_approval())
    assert transition.events[0].payload == {"tool_call_id": "tool"}
    step = transition.state.current_step
    assert step is not None and step.tool_call is not None
    persisted_request = ToolCallRequest(**vars(request))
    restored = replace(
        transition.state,
        current_step=replace(step, tool_call=replace(step.tool_call, request=persisted_request)),
    )
    ready = advance(restored, decide(), resume())
    assert ready.current_step is not None and ready.current_step.tool_call is not None
    assert ready.current_step.tool_call.request == request
    assert advance(ready, finish_tool(), finish_step(), complete()).status is RunStatus.COMPLETED


@pytest.mark.parametrize(
    "arguments_json",
    [
        '["not an object"]',
        '{"n":NaN}',
        '{"n":1,"n":2}',
        '{"n":1,"\\u006e":2}',
        '{"\\ud800":"value"}',
        '{"text":"\\ud800"}',
        '{"nested":' + "[" * 16 + "0" + "]" * 16 + "}",
        '{"text":"' + "é" * 8192 + '"}',
    ],
)
def test_rejects_invalid_or_oversized_tool_request_payload(arguments_json):
    start_step, start_model, finish_model = begin_step()
    running = advance(initial(), submit(), start_step, start_model)
    with pytest.raises(InvalidTransition) as error:
        command = replace(
            finish_model,
            tool_request=ToolCallRequest(
                "tool", "lookup", arguments_json, approval_scope="read", workspace_id="work"
            ),
        )
        TaskStateMachine.apply(running, command)
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize(
    "arguments_json",
    [
        '{"path":"src/nested/main.py"}',
        '{"n":9007199254740993.0,"small":1e-999,"large":1e999}',
        '{"path":"C:\\\\temp\\\\host.txt"}',
        '{"path":"\\u002ftmp\\u002fhost.txt"}',
        '{"nested":' + "[" * 15 + "0" + "]" * 15 + "}",
        '{"path":"' + "é" * 8186 + 'x"}',
    ],
)
def test_tool_request_retains_path_and_exact_boundaries_for_tool_adapter(arguments_json):
    request = ToolCallRequest(
        "tool", "read_file", arguments_json, approval_scope="read", workspace_id="work"
    )
    assert request.arguments_json == arguments_json


@pytest.mark.parametrize(
    "bindings",
    [
        {},
        {"approval_scope": "read"},
        {"workspace_id": "workspace-1"},
        {"approval_scope": "read", "workspace_id": r"C:\secret"},
        {"approval_scope": "/host", "workspace_id": "workspace-1"},
    ],
)
def test_tool_call_projection_requires_an_opaque_approval_binding(bindings):
    with pytest.raises((TypeError, InvalidTransition)):
        ToolCallRequest("tool", "lookup", "{}", **bindings)


@pytest.mark.parametrize("decision", [ToolApproval.APPROVED, ToolApproval.DENIED])
def test_resumed_tool_request_uses_original_journaled_binding(decision):
    request = ToolCallRequest(
        "tool",
        "read_file",
        ' {"number":1e-999}',
        approval_scope="read-v1",
        workspace_id="checkout-1",
    )
    start_step, start_model, finish_model = begin_step()
    running = advance(initial(), submit(), start_step, start_model)
    waiting = advance(running, replace(finish_model, tool_request=request), ask_approval())
    object.__setattr__(request, "approval_scope", "changed-after-capture")

    async def recover():
        journal = InMemoryRunJournal()
        await journal.commit(JournalCommit(waiting), expected_version=0)
        return (await journal.load_head(SESSION)).state

    restored = asyncio.run(recover())
    ready = advance(restored, decide(decision is ToolApproval.APPROVED), resume())
    original = ready.current_step.tool_call.request
    tool = ToolRequest.from_call(
        original,
        run_id=ready.run_id,
        session_id=ready.session_id,
        step_id=ready.current_step.step_id,
        approval=decision,
    )
    assert (tool.approval_scope, tool.workspace_id) == ("read-v1", "checkout-1")
    assert tool.arguments_json == ' {"number":1e-999}'
    for changes in (
        {"approval_scope": "read-v2"},
        {"workspace_id": "checkout-2"},
        {"tool_call_id": "other"},
        {"name": "write_file"},
        {"arguments_json": '{"number":1e-999}'},
        {"original_call": None},
    ):
        with pytest.raises(ValueError):
            replace(tool, **changes)
