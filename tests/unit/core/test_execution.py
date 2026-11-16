from __future__ import annotations

from datetime import datetime, timezone

import pytest
from src.core.execution import (
    ExecutionContext,
    ExecutionLimits,
    FactRecorder,
    LifecycleError,
    LifecycleTracker,
    PayloadLimitError,
)
from src.core.persistence import ExecutionCommitError
from src.core.task import Task
from src.persistence.in_memory import InMemoryTaskRepository
from src.schema.execution import PendingSessionEvent, SessionEventType


def _event(
    event_type: SessionEventType,
    *,
    event_id: str | None = None,
    execution_id: str = "execution-1",
    turn_id: str | None = "turn-1",
    step_id: str | None = None,
    attempt_id: str | None = None,
    tool_call_id: str | None = None,
    outcome: str | None = None,
    payload: dict | None = None,
) -> PendingSessionEvent:
    body = dict(payload or {})
    if outcome is not None:
        body["outcome"] = outcome
    return PendingSessionEvent(
        event_id=event_id or f"event-{event_type.value}-{execution_id}",
        session_id="session-1",
        type=event_type,
        task_id="task-1",
        execution_id=execution_id,
        turn_id=turn_id,
        step_id=step_id,
        attempt_id=attempt_id,
        tool_call_id=tool_call_id,
        occurred_at=datetime(2026, 11, 16, tzinfo=timezone.utc),
        payload=body,
    )


def _completed_history(outcome: str = "completed") -> list[PendingSessionEvent]:
    return [
        _event(SessionEventType.SUBMISSION_ACCEPTED),
        _event(SessionEventType.EXECUTION_STARTED),
        _event(SessionEventType.TURN_STARTED),
        _event(SessionEventType.STEP_STARTED, step_id="step-1"),
        _event(
            SessionEventType.MODEL_ATTEMPT_STARTED,
            step_id="step-1",
            attempt_id="attempt-1",
        ),
        _event(
            SessionEventType.MODEL_ATTEMPT_SETTLED,
            step_id="step-1",
            attempt_id="attempt-1",
            outcome=outcome,
        ),
        _event(SessionEventType.STEP_SETTLED, step_id="step-1", outcome=outcome),
        _event(SessionEventType.TURN_SETTLED, outcome=outcome),
        _event(SessionEventType.SUBMISSION_SETTLED, outcome=outcome),
        _event(SessionEventType.EXECUTION_SETTLED, outcome=outcome),
    ]


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "interrupted"])
def test_lifecycle_accepts_balanced_terminal_histories(outcome):
    tracker = LifecycleTracker()

    tracker.extend(_completed_history(outcome))

    assert tracker.is_settled("execution", "execution-1")
    assert tracker.is_settled("turn", "turn-1")
    assert tracker.is_settled("step", "step-1")
    assert tracker.is_settled("attempt", "attempt-1")


def test_lifecycle_accepts_pause_and_resume_with_a_new_execution():
    tracker = LifecycleTracker()
    tracker.extend(
        [
            _event(SessionEventType.SUBMISSION_ACCEPTED),
            _event(SessionEventType.EXECUTION_STARTED),
            _event(SessionEventType.TURN_STARTED),
            _event(SessionEventType.STEP_STARTED, step_id="step-1"),
            _event(SessionEventType.STEP_PAUSED, step_id="step-1"),
            _event(SessionEventType.TURN_PAUSED),
            _event(SessionEventType.SUBMISSION_PAUSED),
            _event(SessionEventType.EXECUTION_SETTLED, outcome="paused"),
            _event(
                SessionEventType.EXECUTION_STARTED,
                event_id="event-execution-resumed",
                execution_id="execution-2",
            ),
            _event(
                SessionEventType.STEP_RESUMED,
                event_id="event-step-resumed",
                execution_id="execution-2",
                step_id="step-1",
            ),
            _event(
                SessionEventType.TURN_RESUMED,
                event_id="event-turn-resumed",
                execution_id="execution-2",
            ),
            _event(
                SessionEventType.SUBMISSION_RESUMED,
                event_id="event-submission-resumed",
                execution_id="execution-2",
            ),
            _event(
                SessionEventType.STEP_SETTLED,
                event_id="event-step-settled",
                execution_id="execution-2",
                step_id="step-1",
                outcome="completed",
            ),
            _event(
                SessionEventType.TURN_SETTLED,
                event_id="event-turn-settled",
                execution_id="execution-2",
                outcome="completed",
            ),
            _event(
                SessionEventType.SUBMISSION_SETTLED,
                event_id="event-submission-settled",
                execution_id="execution-2",
                outcome="completed",
            ),
            _event(
                SessionEventType.EXECUTION_SETTLED,
                event_id="event-execution-settled",
                execution_id="execution-2",
                outcome="completed",
            ),
        ]
    )

    assert tracker.is_settled("execution", "execution-1")
    assert tracker.is_settled("execution", "execution-2")
    assert tracker.is_settled("turn", "turn-1")


@pytest.mark.parametrize(
    "history, message",
    [
        (
            [
                _event(SessionEventType.TURN_STARTED),
                _event(SessionEventType.TURN_STARTED, event_id="duplicate"),
            ],
            "already started",
        ),
        ([_event(SessionEventType.TURN_SETTLED, outcome="completed")], "was not started"),
        (
            [
                _event(SessionEventType.TURN_STARTED),
                _event(SessionEventType.TURN_SETTLED, outcome="completed"),
                _event(
                    SessionEventType.TURN_SETTLED,
                    event_id="duplicate-settlement",
                    outcome="completed",
                ),
            ],
            "already settled",
        ),
        (
            [
                _event(SessionEventType.TURN_STARTED),
                _event(SessionEventType.TURN_SETTLED, outcome="completed"),
                _event(SessionEventType.STEP_STARTED, step_id="step-1"),
            ],
            "terminal turn",
        ),
        (
            [
                _event(SessionEventType.TURN_STARTED),
                _event(SessionEventType.TURN_RESUMED),
            ],
            "not paused",
        ),
    ],
)
def test_lifecycle_rejects_impossible_transitions(history, message):
    with pytest.raises(LifecycleError, match=message):
        LifecycleTracker().extend(history)


def test_lifecycle_rejects_child_identity_mismatch():
    tracker = LifecycleTracker()
    tracker.apply(_event(SessionEventType.TURN_STARTED))
    tracker.apply(_event(SessionEventType.STEP_STARTED, step_id="step-1"))

    with pytest.raises(LifecycleError, match="hierarchy"):
        tracker.apply(
            _event(
                SessionEventType.STEP_SETTLED,
                event_id="mismatch",
                turn_id="turn-2",
                step_id="step-1",
                outcome="completed",
            )
        )


def test_requested_tool_can_settle_rejected_without_starting():
    tracker = LifecycleTracker()
    tracker.extend(
        [
            _event(SessionEventType.TURN_STARTED),
            _event(SessionEventType.STEP_STARTED, step_id="step-1"),
            _event(
                SessionEventType.TOOL_CALL_REQUESTED,
                step_id="step-1",
                tool_call_id="tool-1",
            ),
            _event(
                SessionEventType.TOOL_CALL_SETTLED,
                event_id="tool-rejected",
                step_id="step-1",
                tool_call_id="tool-1",
                outcome="rejected",
            ),
        ]
    )

    assert tracker.is_settled("tool", "tool-1")


def test_non_rejected_tool_settlement_requires_started_fact():
    tracker = LifecycleTracker()
    tracker.extend(
        [
            _event(SessionEventType.TURN_STARTED),
            _event(SessionEventType.STEP_STARTED, step_id="step-1"),
            _event(
                SessionEventType.TOOL_CALL_REQUESTED,
                step_id="step-1",
                tool_call_id="tool-1",
            ),
        ]
    )

    with pytest.raises(LifecycleError, match="was not started"):
        tracker.apply(
            _event(
                SessionEventType.TOOL_CALL_SETTLED,
                step_id="step-1",
                tool_call_id="tool-1",
                outcome="completed",
            )
        )


def _task() -> Task:
    return Task("work", id="task-1", session_id="session-1", turn_id="turn-1")


def test_execution_context_is_frozen_and_validates_sequence():
    context = ExecutionContext(
        session_id="session-1",
        task_id="task-1",
        execution_id="execution-1",
        turn_id="turn-1",
        expected_sequence=0,
    )

    assert context.expected_sequence == 0
    with pytest.raises(AttributeError):
        context.turn_id = "changed"
    with pytest.raises(ValueError, match="expected_sequence"):
        ExecutionContext(
            session_id="session-1",
            task_id="task-1",
            execution_id="execution-1",
            turn_id="turn-1",
            expected_sequence=-1,
        )


def test_fact_recorder_passes_cas_and_advances_task_only_after_success():
    store = InMemoryTaskRepository()
    recorder = FactRecorder(store)
    task = _task()

    committed = recorder.commit(
        task,
        [
            _event(SessionEventType.EXECUTION_STARTED),
            _event(SessionEventType.TURN_STARTED),
        ],
    )

    assert [event.sequence for event in committed] == [1, 2]
    assert task.last_sequence == 2
    assert store.get("task-1").last_sequence == 2


class _FailingStore(InMemoryTaskRepository):
    def __init__(self, error):
        super().__init__()
        self.error = error
        self.calls = 0

    def commit(self, *args, **kwargs):
        self.calls += 1
        raise self.error


def test_fact_recorder_does_not_advance_sequence_and_preserves_commit_error():
    cause = OSError("disk full")
    error = ExecutionCommitError("commit failed")
    error.__cause__ = cause
    store = _FailingStore(error)
    recorder = FactRecorder(store)
    task = _task()

    with pytest.raises(ExecutionCommitError) as caught:
        recorder.commit(task, [_event(SessionEventType.EXECUTION_STARTED)])

    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert task.last_sequence is None
    assert store.calls == 1


@pytest.mark.parametrize(
    "limits, event_payload, task_text, message",
    [
        (ExecutionLimits(max_event_bytes=120, max_task_snapshot_bytes=10_000), {"x": "y"}, "ok", "event"),
        (ExecutionLimits(max_event_bytes=10_000, max_task_snapshot_bytes=120), {}, "x" * 500, "snapshot"),
    ],
)
def test_fact_recorder_rejects_oversize_payloads_before_store_mutation(
    limits, event_payload, task_text, message
):
    store = InMemoryTaskRepository()
    recorder = FactRecorder(store, limits=limits)
    task = Task(task_text, id="task-1", session_id="session-1", turn_id="turn-1")

    with pytest.raises(PayloadLimitError, match=message):
        recorder.commit(
            task,
            [_event(SessionEventType.EXECUTION_STARTED, payload=event_payload)],
        )

    assert store.get("task-1") is None
    assert store.events("session-1") == []
    assert task.last_sequence is None


def test_fact_recorder_rejects_invalid_lifecycle_before_store_mutation():
    store = InMemoryTaskRepository()
    recorder = FactRecorder(store)
    task = _task()

    with pytest.raises(LifecycleError, match="was not started"):
        recorder.commit(task, [_event(SessionEventType.TURN_SETTLED, outcome="failed")])

    assert store.get("task-1") is None
