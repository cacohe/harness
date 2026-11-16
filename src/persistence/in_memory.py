from __future__ import annotations

import threading
from collections.abc import Sequence

from src.core.persistence import (
    ExecutionRecord,
    SequenceConflictError,
    _assign_events,
    _commit_session_id,
    _decode_event,
    _decode_submission,
    _decode_task,
    _encode_event,
    _encode_submission,
    _encode_task,
    _validate_expected_sequence,
    _validate_submission_update,
)
from src.core.task import Task
from src.schema.execution import PendingSessionEvent, SessionEvent, SubmissionRecord


class InMemoryTaskRepository:
    """Keep atomic task snapshots and execution facts for one process."""

    def __init__(self) -> None:
        self._tasks: dict[str, str] = {}
        self._events: dict[str, list[str]] = {}
        self._submissions: dict[tuple[str, str], str] = {}
        self._event_ids: set[str] = set()
        self._lock = threading.RLock()

    def save(self, task: Task) -> None:
        encoded = _encode_task(task)
        with self._lock:
            self._tasks[task.id] = encoded

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            encoded = self._tasks.get(task_id)
            return None if encoded is None else _decode_task(encoded)

    def commit(
        self,
        task: Task,
        events: Sequence[PendingSessionEvent],
        *,
        expected_sequence: int | None,
        submission: SubmissionRecord | None = None,
    ) -> list[SessionEvent]:
        expected = _validate_expected_sequence(expected_sequence, allow_none=False)
        assert expected is not None
        pending = [PendingSessionEvent.model_validate(event.model_dump()) for event in events]
        incoming_submission = (
            None if submission is None else _decode_submission(_encode_submission(submission))
        )
        session_id = _commit_session_id(task, pending, incoming_submission)
        with self._lock:
            existing_events = self._events.get(session_id, [])
            current = len(existing_events)
            if current != expected:
                raise SequenceConflictError(
                    f"expected session sequence {expected}, found {current}"
                )
            duplicate_ids = self._event_ids.intersection(event.event_id for event in pending)
            if duplicate_ids or len({event.event_id for event in pending}) != len(pending):
                raise SequenceConflictError("event identity is already committed")
            existing_submission = (
                self.submission(session_id, incoming_submission.message_id)
                if incoming_submission is not None
                else None
            )
            incoming_submission = _validate_submission_update(
                existing_submission, incoming_submission
            )
            assigned = _assign_events(pending, current)
            snapshot = Task.from_dict(task.to_dict())
            snapshot.last_sequence = current + len(assigned)
            encoded_task = _encode_task(snapshot)
            encoded_events = [_encode_event(event) for event in assigned]
            encoded_submission = (
                None if incoming_submission is None else _encode_submission(incoming_submission)
            )

            self._tasks[snapshot.id] = encoded_task
            self._events.setdefault(session_id, []).extend(encoded_events)
            self._event_ids.update(event.event_id for event in assigned)
            if incoming_submission is not None and encoded_submission is not None:
                self._submissions[(session_id, incoming_submission.message_id)] = encoded_submission
            return [_decode_event(encoded) for encoded in encoded_events]

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
