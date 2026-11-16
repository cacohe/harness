from datetime import datetime, timedelta, timezone

import pytest
from src.core.persistence import (
    ExecutionRecord,
    SequenceConflictError,
    SubmissionConflictError,
)
from src.core.task import Task
from src.persistence.in_memory import InMemoryTaskRepository
from src.schema.execution import (
    PendingSessionEvent,
    SessionEventType,
    SubmissionRecord,
    SubmissionState,
)
from src.schema.task import TaskResult, TaskRunningReason


def _event(event_id: str, *, session_id: str = "session-1") -> PendingSessionEvent:
    return PendingSessionEvent(
        event_id=event_id,
        session_id=session_id,
        type=SessionEventType.TURN_STARTED,
        task_id="task-1",
        execution_id="execution-1",
        turn_id="turn-1",
        payload={},
    )


def _submission(
    *,
    digest: str = "a" * 64,
    state: SubmissionState = SubmissionState.ACCEPTED,
    timestamp: datetime | None = None,
) -> SubmissionRecord:
    now = timestamp or datetime.now(timezone.utc)
    return SubmissionRecord(
        session_id="session-1",
        message_id="message-1",
        digest_algorithm="sha256-raw-utf8-v1",
        message_digest=digest,
        task_id="task-1",
        turn_id="turn-1",
        state=state,
        created_at=now,
        updated_at=now,
    )


def _task() -> Task:
    return Task(
        "work",
        id="task-1",
        session_id="session-1",
        message_id="message-1",
        turn_id="turn-1",
    )


def test_commit_assigns_contiguous_sequences_and_saves_independent_snapshots():
    repository = InMemoryTaskRepository()
    task = _task()

    first = repository.commit(
        task,
        [_event("event-1"), _event("event-2")],
        expected_sequence=0,
        submission=_submission(),
    )

    assert [event.sequence for event in first] == [1, 2]
    assert task.last_sequence is None
    restored = repository.get(task.id)
    assert restored is not None
    assert restored.last_sequence == 2
    stored_submission = repository.submission("session-1", "message-1")
    assert stored_submission is not None
    assert stored_submission.message_digest == "a" * 64
    assert stored_submission.state is SubmissionState.ACCEPTED

    restored.original_input = "mutated read"
    first[0].payload["mutated"] = True
    assert repository.get(task.id).original_input == "work"
    assert repository.events("session-1")[0].payload == {}


def test_commit_filters_events_after_sequence_and_returns_independent_events():
    repository = InMemoryTaskRepository()
    repository.commit(_task(), [_event("event-1"), _event("event-2")], expected_sequence=0)

    events = repository.events("session-1", after_sequence=1)

    assert [event.event_id for event in events] == ["event-2"]
    events[0].payload["changed"] = True
    assert repository.events("session-1", after_sequence=1)[0].payload == {}


def test_stale_sequence_conflict_does_not_change_task_events_or_submission():
    repository = InMemoryTaskRepository()
    repository.commit(_task(), [_event("event-1")], expected_sequence=0)
    changed = _task()
    changed.original_input = "must not persist"

    with pytest.raises(SequenceConflictError):
        repository.commit(
            changed,
            [_event("event-2")],
            expected_sequence=0,
            submission=_submission(),
        )

    assert repository.get("task-1").original_input == "work"
    assert [event.event_id for event in repository.events("session-1")] == ["event-1"]
    assert repository.submission("session-1", "message-1") is None


def test_submission_conflict_does_not_partially_commit_other_changes():
    repository = InMemoryTaskRepository()
    repository.commit(_task(), [_event("event-1")], expected_sequence=0, submission=_submission())
    changed = _task()
    changed.original_input = "must not persist"

    with pytest.raises(SubmissionConflictError):
        repository.commit(
            changed,
            [_event("event-2")],
            expected_sequence=1,
            submission=_submission(digest="b" * 64),
        )

    assert repository.get("task-1").original_input == "work"
    assert [event.event_id for event in repository.events("session-1")] == ["event-1"]
    assert repository.submission("session-1", "message-1").message_digest == "a" * 64


def test_matching_submission_can_advance_state_without_changing_identity():
    repository = InMemoryTaskRepository()
    task = _task()
    created_at = datetime(2026, 11, 16, tzinfo=timezone.utc)
    repository.commit(
        task,
        [_event("event-1")],
        expected_sequence=0,
        submission=_submission(timestamp=created_at),
    )

    repository.commit(
        task,
        [_event("event-2")],
        expected_sequence=1,
        submission=_submission(
            state=SubmissionState.COMPLETED,
            timestamp=created_at + timedelta(seconds=1),
        ),
    )

    stored = repository.submission("session-1", "message-1")
    assert stored.state is SubmissionState.COMPLETED
    assert stored.created_at == created_at
    assert stored.updated_at == created_at + timedelta(seconds=1)


def test_unfinished_returns_only_independent_nonterminal_execution_records():
    repository = InMemoryTaskRepository()
    running = _task()
    running.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    repository.commit(running, [_event("event-1")], expected_sequence=0)
    completed = Task(
        "done",
        id="task-2",
        session_id="session-2",
        state="completed",
        result=TaskResult(output="done"),
    )
    repository.commit(
        completed,
        [_event("event-2", session_id="session-2").model_copy(update={"task_id": "task-2"})],
        expected_sequence=0,
    )

    records = repository.unfinished()

    assert len(records) == 1
    assert isinstance(records[0], ExecutionRecord)
    assert records[0].session_id == "session-1"
    assert records[0].last_sequence == 1
    assert records[0].task.state.value == "running"
    records[0].task.original_input = "changed"
    assert repository.get("task-1").original_input == "work"


def test_real_execution_store_requires_an_integer_expected_sequence():
    repository = InMemoryTaskRepository()

    with pytest.raises(TypeError, match="expected_sequence"):
        repository.commit(_task(), [_event("event-1")], expected_sequence=None)
