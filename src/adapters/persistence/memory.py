from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from threading import Lock
from typing import TypeVar

from src.core.errors import RunErrorCode
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalSnapshot,
    SubmissionRecord,
    TranscriptEntry,
)
from src.core.state import RunState


@dataclass
class _Session:
    version: int = 0
    sequence: int = 0
    state: RunState | None = None
    runs: dict[RunId, RunState] = field(default_factory=dict)
    events: list[JournalEvent] = field(default_factory=list)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    summaries: dict[tuple[SessionId, int, str, ProfileId], ConversationSummaryCheckpoint] = field(
        default_factory=dict
    )
    submissions: dict[SubmissionId, tuple[JournalCommit, JournalSnapshot]] = field(
        default_factory=dict
    )


class InMemoryRunJournal:
    """An in-process journal with atomic writes across all Session indexes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[SessionId, _Session] = {}
        self._event_ids: set[EventId] = set()

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        with self._lock:
            if (
                type(expected_version) is not int
                or expected_version < 0
                or type(batch) is not JournalCommit
                or type(batch.state) is not RunState
            ):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            state = batch.state
            session = self._sessions.get(state.session_id)
            if session is None:
                session = _Session()

            _validate_members(batch, session.sequence)
            accepted = session.submissions.get(state.submission_id)
            if accepted is not None:
                original, receipt = accepted
                if batch.submission is not None:
                    if batch != original:
                        raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                    return _copy(receipt)
                if receipt.state is not None and receipt.state.run_id != state.run_id:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            if session.version != expected_version:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            previous = session.runs.get(state.run_id)
            if previous is not None and (
                previous.submission_id != state.submission_id
                or previous.profile_id != state.profile_id
                or state.version < previous.version
            ):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            event_ids = {event.event_id for event in batch.events}
            if len(event_ids) != len(batch.events) or not self._event_ids.isdisjoint(event_ids):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            batch = _copy(batch)
            cursor = session.sequence
            events = tuple(
                JournalEvent(sequence, event)
                for sequence, event in enumerate(batch.events, start=cursor + 1)
            )
            cursor += len(events)
            transcript = tuple(
                replace(entry, sequence=sequence)
                for sequence, entry in enumerate(batch.transcript, start=cursor + 1)
            )
            receipt = JournalSnapshot(
                session_id=batch.state.session_id,
                version=session.version + 1,
                sequence=cursor + len(transcript),
                state=batch.state,
                events=events,
                transcript=transcript,
                summaries=(batch.summary,) if batch.summary is not None else (),
                submission=batch.submission,
            )
            result = _copy(receipt)

            # All checks and copies complete before updates to the journal and its indexes.
            session.version = receipt.version
            session.sequence = receipt.sequence
            session.state = batch.state
            session.runs[batch.state.run_id] = batch.state
            session.events.extend(events)
            session.transcript.extend(transcript)
            if batch.summary is not None:
                session.summaries[batch.summary.key] = batch.summary
            if batch.submission is not None:
                session.submissions[batch.submission.submission_id] = batch, receipt
            self._event_ids.update(event.event_id for event in batch.events)
            self._sessions[receipt.session_id] = session
            return result

    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        if (
            type(session_id) is not SessionId
            or type(after_sequence) is not int
            or after_sequence < 0
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return JournalSnapshot(session_id, 0, 0)

            event_start = bisect_right(
                session.events, after_sequence, key=lambda entry: entry.sequence
            )
            transcript_start = bisect_right(
                session.transcript, after_sequence, key=lambda entry: entry.sequence
            )
            return _copy(
                JournalSnapshot(
                    session_id,
                    session.version,
                    session.sequence,
                    session.state,
                    tuple(session.events[event_start:]),
                    tuple(session.transcript[transcript_start:]),
                    tuple(session.summaries.values()),
                )
            )

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionRecord | None:
        if type(session_id) is not SessionId or type(submission_id) is not SubmissionId:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        with self._lock:
            session = self._sessions.get(session_id)
            accepted = session.submissions.get(submission_id) if session is not None else None
            return _copy(accepted[0].submission) if accepted is not None else None


def _validate_members(batch: JournalCommit, sequence: int) -> None:
    state = batch.state
    if type(batch.events) is not tuple or type(batch.transcript) is not tuple:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    for event in batch.events:
        if (
            type(event) is not DomainEvent
            or type(event.event_id) is not EventId
            or type(event.type) is not DomainEventType
            or type(event.timestamp) is not datetime
            or event.session_id != state.session_id
            or event.run_id != state.run_id
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    submission = batch.submission
    if submission is not None and (
        type(submission) is not SubmissionRecord
        or submission.session_id != state.session_id
        or submission.run_id != state.run_id
        or submission.submission_id != state.submission_id
        or not _is_text(submission.digest)
        or not submission.digest
    ):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    for entry in batch.transcript:
        if (
            type(entry) is not TranscriptEntry
            or entry.session_id != state.session_id
            or entry.run_id != state.run_id
            or entry.submission_id != state.submission_id
            or type(entry.sequence) is not int
            or entry.sequence != 0
            or not _is_text(entry.role)
            or not entry.role
            or not _is_text(entry.content)
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    summary = batch.summary
    next_sequence = sequence + len(batch.events) + len(batch.transcript)
    if summary is not None and (
        type(summary) is not ConversationSummaryCheckpoint
        or summary.session_id != state.session_id
        or type(summary.profile_id) is not ProfileId
        or type(summary.covered_from_sequence) is not int
        or type(summary.covered_through_sequence) is not int
        or not 1 <= summary.covered_from_sequence <= summary.covered_through_sequence
        or summary.covered_through_sequence > next_sequence
        or not _is_text(summary.policy_version)
        or not summary.policy_version
        or not _is_text(summary.summary)
        or not summary.summary
    ):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)


def _is_text(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


_T = TypeVar("_T")


def _copy(value: _T) -> _T:
    try:
        return deepcopy(value)
    except Exception:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
