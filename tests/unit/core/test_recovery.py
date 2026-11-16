import pytest
from src.core.recovery import ExecutionRecovery, RecoveryError
from src.core.task import Task
from src.persistence.in_memory import InMemoryTaskRepository
from src.schema.execution import PendingSessionEvent, SessionEventType
from src.schema.task import TaskPauseReason, TaskRunningReason


def _event(event_type, event_id, **identities):
    payload = identities.pop("payload", {})
    return PendingSessionEvent(
        event_id=event_id,
        session_id="session-1",
        type=event_type,
        task_id="task-1",
        execution_id="execution-1",
        turn_id="turn-1",
        payload=payload,
        **identities,
    )


def _interrupted_store():
    store = InMemoryTaskRepository()
    task = Task(
        "work",
        id="task-1",
        session_id="session-1",
        message_id="message-1",
        turn_id="turn-1",
    )
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    events = [
        _event(SessionEventType.SUBMISSION_ACCEPTED, "submission"),
        _event(SessionEventType.EXECUTION_STARTED, "execution"),
        _event(SessionEventType.TURN_STARTED, "turn"),
        _event(SessionEventType.STEP_STARTED, "step", step_id="step-1"),
        _event(
            SessionEventType.MODEL_ATTEMPT_STARTED,
            "attempt",
            step_id="step-1",
            attempt_id="attempt-1",
        ),
    ]
    store.commit(task, events, expected_sequence=0)
    return store


def test_recovery_appends_minimal_reverse_order_interrupted_settlements():
    store = _interrupted_store()

    recovered = ExecutionRecovery(store).recover_session("session-1")

    assert [event.type for event in recovered] == [
        SessionEventType.MODEL_ATTEMPT_SETTLED,
        SessionEventType.STEP_SETTLED,
        SessionEventType.TURN_SETTLED,
        SessionEventType.SUBMISSION_SETTLED,
        SessionEventType.EXECUTION_SETTLED,
    ]
    assert all(event.payload["outcome"] == "interrupted" for event in recovered)
    restored = store.get("task-1")
    assert restored.is_failed()
    assert restored.state_reason.value == "interrupted"
    assert ExecutionRecovery(store).recover_session("session-1") == []


def test_recovery_preserves_paused_approval_without_appending_facts():
    store = InMemoryTaskRepository()
    task = Task("work", id="task-1", session_id="session-1", turn_id="turn-1")
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    task.transition_to_paused(TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL)
    events = [
        _event(SessionEventType.SUBMISSION_ACCEPTED, "submission"),
        _event(SessionEventType.EXECUTION_STARTED, "execution"),
        _event(SessionEventType.TURN_STARTED, "turn"),
        _event(SessionEventType.STEP_STARTED, "step", step_id="step-1"),
        _event(SessionEventType.STEP_PAUSED, "step-paused", step_id="step-1"),
        _event(SessionEventType.TURN_PAUSED, "turn-paused"),
        _event(SessionEventType.SUBMISSION_PAUSED, "submission-paused"),
        _event(
            SessionEventType.EXECUTION_SETTLED,
            "execution-paused",
            payload={"outcome": "paused"},
        ),
    ]
    store.commit(task, events, expected_sequence=0)
    before = store.events("session-1")

    assert ExecutionRecovery(store).recover_session("session-1") == []
    assert store.events("session-1") == before
    assert store.get("task-1").is_paused()


def test_recovery_rejects_sequence_gap_without_modifying_storage():
    class GappedStore(InMemoryTaskRepository):
        def events(self, session_id, *, after_sequence=0):
            events = super().events(session_id, after_sequence=after_sequence)
            if len(events) > 1:
                events[1] = events[1].model_copy(update={"sequence": 3})
            return events

    base = _interrupted_store()
    store = GappedStore()
    store._tasks = base._tasks
    store._events = base._events
    store._event_ids = base._event_ids

    with pytest.raises(RecoveryError, match="gap"):
        ExecutionRecovery(store).recover_session("session-1")

    assert len(base.events("session-1")) == 5


def test_recovery_rejects_task_turn_disagreement_without_writes():
    store = _interrupted_store()
    stored = store.get("task-1")
    stored.turn_id = "different-turn"
    store.save(stored)

    with pytest.raises(RecoveryError, match="turn identity"):
        ExecutionRecovery(store).recover_session("session-1")

    assert len(store.events("session-1")) == 5
