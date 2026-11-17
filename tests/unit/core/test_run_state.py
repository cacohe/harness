from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from itertools import product

import pytest
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
)
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.state import InvalidTransition, RunState, RunStatus, TaskStateMachine

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


def begin_step(step_id: str = "step", tool_call_id: str | None = "tool") -> tuple[CoreCommand, ...]:
    return (
        StartStep(RUN, SESSION, step_id, context=context()),
        StartModelCall(RUN, SESSION, step_id, "model", context=context()),
        FinishModelCall(
            RUN, SESSION, step_id, "model", "model result", tool_call_id, context=context()
        ),
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


def test_event_metadata_must_match_the_events_actually_emitted():
    running = advance(initial(), submit())
    with pytest.raises(InvalidTransition):
        advance(initial(), replace(submit(), context=context()))
    with pytest.raises(InvalidTransition):
        advance(running, StartStep(RUN, SESSION, "step", context=context("unexpected")))
    with pytest.raises(InvalidTransition):
        advance(
            running,
            replace(complete(), context=CommandContext(NOW, 0, (EventId("a"), EventId("b")))),
        )


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
