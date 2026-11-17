from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from threading import Lock
from typing import TypeAlias, TypeVar

from src.core.commands import RunLimits, ToolCallRequest
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType, _is_canonical_timestamp
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
from src.core.state import (
    ApprovalState,
    ModelCallState,
    RunState,
    RunStatus,
    StepState,
    ToolCallState,
)

_IntentValue: TypeAlias = "str | int | None | tuple[_IntentValue, ...]"
_RECORD_TYPES = (
    JournalCommit,
    RunState,
    RunLimits,
    RunFailure,
    ApprovalState,
    ModelCallState,
    StepState,
    ToolCallState,
    ToolCallRequest,
    DomainEvent,
    SubmissionRecord,
    TranscriptEntry,
    ConversationSummaryCheckpoint,
)


@dataclass
class _Session:
    version: int = 0
    sequence: int = 0
    state: RunState | None = None
    runs: dict[str, RunState] = field(default_factory=dict)
    events: list[JournalEvent] = field(default_factory=list)
    transcript: list[TranscriptEntry] = field(default_factory=list)
    summaries: dict[tuple[str, int, str, str], ConversationSummaryCheckpoint] = field(
        default_factory=dict
    )
    submissions: dict[str, tuple[_IntentValue, JournalSnapshot]] = field(default_factory=dict)


class InMemoryRunJournal:
    """An in-process journal with atomic writes across all Session indexes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, _Session] = {}
        self._event_ids: set[str] = set()

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        with self._lock:
            if (
                type(expected_version) is not int
                or expected_version < 0
                or type(batch) is not JournalCommit
                or type(batch.state) is not RunState
            ):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            intent = _validated_intent(batch)
            state = batch.state
            session_key = state.session_id.value
            run_key = state.run_id.value
            submission_key = state.submission_id.value
            session = self._sessions.get(session_key)
            if session is None:
                session = _Session()

            _validate_members(batch, session.sequence)
            accepted = session.submissions.get(submission_key)
            if accepted is not None:
                original, receipt = accepted
                if batch.submission is not None:
                    if intent != original:
                        raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                    return _copy(receipt)
                if receipt.state is not None and receipt.state.run_id != state.run_id:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            if session.version != expected_version:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            previous = session.runs.get(run_key)
            if previous is not None and (
                previous.submission_id != state.submission_id
                or previous.profile_id != state.profile_id
                or state.version < previous.version
            ):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            event_ids = {event.event_id.value for event in batch.events}
            if len(event_ids) != len(batch.events) or not self._event_ids.isdisjoint(event_ids):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            summary = batch.summary
            summary_key = (
                (
                    session_key,
                    summary.covered_through_sequence,
                    summary.policy_version,
                    summary.profile_id.value,
                )
                if summary is not None
                else None
            )
            batch = _copy(batch)
            summary = batch.summary
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
                summaries=(summary,) if summary is not None else (),
                submission=batch.submission,
            )
            result = _copy(receipt)

            # All checks and copies complete before updates to the journal and its indexes.
            session.version = receipt.version
            session.sequence = receipt.sequence
            session.state = batch.state
            session.runs[run_key] = batch.state
            session.events.extend(events)
            session.transcript.extend(transcript)
            if summary_key is not None and summary is not None:
                session.summaries[summary_key] = summary
            if batch.submission is not None:
                session.submissions[submission_key] = intent, receipt
            self._event_ids.update(event_ids)
            self._sessions[session_key] = session
            return result

    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        if (
            type(session_id) is not SessionId
            or type(session_id.value) is not str
            or type(after_sequence) is not int
            or after_sequence < 0
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        with self._lock:
            session = self._sessions.get(session_id.value)
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
        if (
            type(session_id) is not SessionId
            or type(session_id.value) is not str
            or type(submission_id) is not SubmissionId
            or type(submission_id.value) is not str
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        with self._lock:
            session = self._sessions.get(session_id.value)
            accepted = session.submissions.get(submission_id.value) if session is not None else None
            return _copy(accepted[1].submission) if accepted is not None else None


def _validated_intent(batch: JournalCommit) -> _IntentValue:
    try:
        intent = _intent_value(batch)
        if type(batch.state.limits) is not RunLimits:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        RunLimits.__post_init__(batch.state.limits)
        RunState.__post_init__(batch.state)
        return intent
    except (AttributeError, TypeError, ValueError, RecursionError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


def _intent_value(value: object) -> _IntentValue:
    """Capture records as immutable, typed persistence values without custom equality."""
    if value is None or type(value) is int or type(value) is str:
        return value
    if type(value) is bool:
        return ("bool", int(value))
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is datetime and _is_canonical_timestamp(value):
        return ("datetime", value.isoformat(timespec="microseconds"))
    if type(value) is RunStatus or type(value) is RunErrorCode or type(value) is DomainEventType:
        return (type(value).__name__, value.value)
    if (
        type(value) is SessionId
        or type(value) is RunId
        or type(value) is SubmissionId
        or type(value) is ProfileId
        or type(value) is EventId
    ) and type(value.value) is str:
        return (type(value).__name__, value.value)
    if type(value) is tuple:
        return tuple(_intent_value(item) for item in value)
    if type(value) is frozenset and all(type(item) is str for item in value):
        return ("frozenset", *sorted(value))
    if any(type(value) is record_type for record_type in _RECORD_TYPES) and is_dataclass(value):
        return (
            type(value).__name__,
            *(_intent_value(getattr(value, member.name)) for member in fields(value)),
        )
    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)


def _validate_members(batch: JournalCommit, sequence: int) -> None:
    state = batch.state
    if type(batch.events) is not tuple or type(batch.transcript) is not tuple:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    for event in batch.events:
        if (
            type(event) is not DomainEvent
            or type(event.event_id) is not EventId
            or type(event.type) is not DomainEventType
            or not _is_canonical_timestamp(event.timestamp)
            or type(event._payload_json) is not str
            or type(event.session_id) is not SessionId
            or type(event.run_id) is not RunId
            or event.session_id != state.session_id
            or event.run_id != state.run_id
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    submission = batch.submission
    if submission is not None and (
        type(submission) is not SubmissionRecord
        or type(submission.session_id) is not SessionId
        or type(submission.run_id) is not RunId
        or type(submission.submission_id) is not SubmissionId
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
            or type(entry.session_id) is not SessionId
            or type(entry.run_id) is not RunId
            or type(entry.submission_id) is not SubmissionId
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
        or type(summary.session_id) is not SessionId
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
