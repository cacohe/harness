from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.core.task import Task
from src.schema.execution import PendingSessionEvent, SessionEvent, SubmissionRecord


@runtime_checkable
class TaskRepository(Protocol):
    def save(self, task: Task) -> None: ...

    def get(self, task_id: str) -> Task | None: ...


class NoOpTaskRepository:
    """Explicitly disable task persistence."""

    def save(self, task: Task) -> None:
        return None

    def get(self, task_id: str) -> Task | None:
        return None


class ExecutionCommitError(RuntimeError):
    """An execution fact batch could not be committed atomically."""


class SequenceConflictError(ExecutionCommitError):
    """A session writer used a stale event sequence."""


class SubmissionConflictError(ExecutionCommitError):
    """One message identity was reused for different content or ownership."""


@dataclass(frozen=True)
class ExecutionRecord:
    session_id: str
    task: Task
    last_sequence: int


@runtime_checkable
class ExecutionStore(Protocol):
    def commit(
        self,
        task: Task,
        events: Sequence[PendingSessionEvent],
        *,
        expected_sequence: int | None,
        submission: SubmissionRecord | None = None,
    ) -> list[SessionEvent]: ...

    def events(self, session_id: str, *, after_sequence: int = 0) -> list[SessionEvent]: ...

    def submission(self, session_id: str, message_id: str) -> SubmissionRecord | None: ...

    def unfinished(self, session_id: str | None = None) -> list[ExecutionRecord]: ...


def _encode_task(task: Task) -> str:
    encoded = json.dumps(
        task.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    Task.from_dict(json.loads(encoded))
    return encoded


def _decode_task(encoded: str) -> Task:
    return Task.from_dict(json.loads(encoded))


def _encode_event(event: SessionEvent) -> str:
    return event.model_dump_json()


def _decode_event(encoded: str) -> SessionEvent:
    return SessionEvent.model_validate_json(encoded)


def _encode_submission(submission: SubmissionRecord) -> str:
    return submission.model_dump_json()


def _decode_submission(encoded: str) -> SubmissionRecord:
    return SubmissionRecord.model_validate_json(encoded)


def _validate_expected_sequence(value: int | None, *, allow_none: bool) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("expected_sequence must be a non-negative integer")
    return value


def _validate_submission_update(
    existing: SubmissionRecord | None, incoming: SubmissionRecord | None
) -> SubmissionRecord | None:
    if existing is None or incoming is None:
        return incoming
    identity = (
        "session_id",
        "message_id",
        "digest_algorithm",
        "message_digest",
        "task_id",
        "turn_id",
    )
    if any(getattr(existing, field) != getattr(incoming, field) for field in identity):
        raise SubmissionConflictError("message identity conflicts with the existing submission")
    return SubmissionRecord.model_validate(
        {**incoming.model_dump(), "created_at": existing.created_at}
    )


class SnapshotExecutionStore:
    """Add process-local facts to one legacy TaskRepository."""

    def __init__(self, repository: TaskRepository) -> None:
        self.repository = repository
        self._events: dict[str, list[str]] = {}
        self._submissions: dict[tuple[str, str], str] = {}
        self._tasks: dict[str, str] = {}
        self._lock = threading.RLock()

    def commit(
        self,
        task: Task,
        events: Sequence[PendingSessionEvent],
        *,
        expected_sequence: int | None,
        submission: SubmissionRecord | None = None,
    ) -> list[SessionEvent]:
        expected = _validate_expected_sequence(expected_sequence, allow_none=True)
        pending = [PendingSessionEvent.model_validate(event.model_dump()) for event in events]
        incoming_submission = (
            None if submission is None else _decode_submission(_encode_submission(submission))
        )
        session_id = _commit_session_id(task, pending, incoming_submission)
        with self._lock:
            current = len(self._events.get(session_id, []))
            if expected is not None and expected != current:
                raise SequenceConflictError(
                    f"expected session sequence {expected}, found {current}"
                )
            existing = (
                self.submission(session_id, incoming_submission.message_id)
                if incoming_submission
                else None
            )
            incoming_submission = _validate_submission_update(existing, incoming_submission)
            assigned = _assign_events(pending, current)
            snapshot = Task.from_dict(task.to_dict())
            snapshot.last_sequence = current + len(assigned)
            encoded_task = _encode_task(snapshot)
            self.repository.save(snapshot)
            self._tasks[snapshot.id] = encoded_task
            self._events.setdefault(session_id, []).extend(_encode_event(item) for item in assigned)
            if incoming_submission is not None:
                key = (incoming_submission.session_id, incoming_submission.message_id)
                self._submissions[key] = _encode_submission(incoming_submission)
            return [_decode_event(_encode_event(item)) for item in assigned]

    def events(self, session_id: str, *, after_sequence: int = 0) -> list[SessionEvent]:
        _validate_expected_sequence(after_sequence, allow_none=False)
        with self._lock:
            result = []
            for encoded in self._events.get(session_id, []):
                event = _decode_event(encoded)
                if event.sequence > after_sequence:
                    result.append(event)
            return result

    def submission(self, session_id: str, message_id: str) -> SubmissionRecord | None:
        with self._lock:
            encoded = self._submissions.get((session_id, message_id))
            return None if encoded is None else _decode_submission(encoded)

    def unfinished(self, session_id: str | None = None) -> list[ExecutionRecord]:
        with self._lock:
            records = []
            for encoded in self._tasks.values():
                task = _decode_task(encoded)
                if task.is_finished() or task.session_id is None:
                    continue
                if session_id is not None and task.session_id != session_id:
                    continue
                records.append(ExecutionRecord(task.session_id, task, task.last_sequence or 0))
            return records


def _commit_session_id(
    task: Task,
    events: Sequence[PendingSessionEvent],
    submission: SubmissionRecord | None,
) -> str:
    candidates = [event.session_id for event in events]
    if task.session_id is not None:
        candidates.append(task.session_id)
    if submission is not None:
        candidates.append(submission.session_id)
    if not candidates or any(value != candidates[0] for value in candidates):
        raise ExecutionCommitError("commit requires one consistent session identity")
    if any(event.task_id != task.id for event in events):
        raise ExecutionCommitError("event task identity does not match snapshot")
    if submission is not None and submission.task_id != task.id:
        raise ExecutionCommitError("submission task identity does not match snapshot")
    return candidates[0]


def _assign_events(
    events: Sequence[PendingSessionEvent], current_sequence: int
) -> list[SessionEvent]:
    return [
        SessionEvent(sequence=current_sequence + index, **event.model_dump())
        for index, event in enumerate(events, start=1)
    ]
