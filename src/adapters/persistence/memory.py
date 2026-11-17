import json
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from threading import Lock
from typing import Any, TypeAlias, TypeVar

from src.core.artifacts import ArtifactRef, ExternalizedContent, snapshot_artifact_ref
from src.core.commands import RunLimits, ToolCallRequest
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType, _is_canonical_timestamp
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalHead,
    JournalSnapshot,
    SubmissionAcceptance,
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

_IntentValue: TypeAlias = tuple[str, object]
_ID_TYPES = (EventId, ProfileId, RunId, SessionId, SubmissionId)
_RECORD_TYPES = (
    JournalCommit,
    JournalSnapshot,
    JournalEvent,
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


@dataclass(frozen=True)
class _Session:
    version: int = 0
    sequence: int = 0
    state: RunState | None = None
    runs: Mapping[str, RunState] = field(default_factory=dict)
    events: tuple[JournalEvent, ...] = ()
    transcript: tuple[TranscriptEntry, ...] = ()
    summaries: Mapping[tuple[str, int, str, str], ConversationSummaryCheckpoint] = field(
        default_factory=dict
    )
    submissions: Mapping[str, tuple[_IntentValue, JournalSnapshot]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Store:
    sessions: Mapping[str, _Session] = field(default_factory=dict)
    event_ids: frozenset[str] = frozenset()


class InMemoryRunJournal:
    """An in-process journal with atomic writes across all Session indexes."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._store = _Store()

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        with self._lock:
            if (
                type(expected_version) is not int
                or expected_version < 0
                or type(batch) is not JournalCommit
            ):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            batch = _canonical_copy(batch)
            state = batch.state
            if type(state) is not RunState:
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            store = self._store
            session_key = state.session_id.value
            run_key = state.run_id.value
            submission_key = state.submission_id.value
            session = store.sessions.get(session_key)
            if session is None:
                session = _Session()

            intent = _intent_value(batch) if batch.submission is not None else None
            accepted = session.submissions.get(submission_key)
            if accepted is not None:
                original, receipt = accepted
                if intent is not None:
                    if intent != original:
                        raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                    return _canonical_copy(receipt)
                if receipt.state is not None and receipt.state.run_id != state.run_id:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            if session.version != expected_version:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            _validate_members(batch, session.sequence, expected_version)
            previous = session.runs.get(run_key)
            if previous is not None and (
                previous.submission_id != state.submission_id
                or previous.profile_id != state.profile_id
                or state.version < previous.version
            ):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

            event_ids = {event.event_id.value for event in batch.events}
            if len(event_ids) != len(batch.events) or not store.event_ids.isdisjoint(event_ids):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

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
            result = _canonical_copy(receipt)
            summaries = session.summaries
            if summary is not None:
                summary_key = (
                    session_key,
                    summary.covered_through_sequence,
                    summary.policy_version,
                    summary.profile_id.value,
                )
                existing_summary = summaries.get(summary_key)
                if existing_summary is not None and existing_summary != summary:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
                if existing_summary is None:
                    summaries = {**summaries, summary_key: summary}
            submissions = session.submissions
            if intent is not None:
                submissions = {**submissions, submission_key: (intent, receipt)}
            next_session = _Session(
                version=receipt.version,
                sequence=receipt.sequence,
                state=state,
                runs={**session.runs, run_key: state},
                events=session.events + events,
                transcript=session.transcript + transcript,
                summaries=summaries,
                submissions=submissions,
            )
            self._store = _Store(
                sessions={**store.sessions, session_key: next_session},
                event_ids=store.event_ids.union(event_ids),
            )
            return result

    async def load_head(self, session_id: SessionId) -> JournalHead:
        if type(session_id) is not SessionId:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        session_id = _canonical_copy(session_id)
        with self._lock:
            session = self._store.sessions.get(session_id.value)
        if session is None:
            return JournalHead(0, 0)
        return JournalHead(session.version, session.sequence, _canonical_copy(session.state))

    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        if (
            type(session_id) is not SessionId
            or type(after_sequence) is not int
            or after_sequence < 0
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        session_id = _canonical_copy(session_id)
        with self._lock:
            store = self._store

        session = store.sessions.get(session_id.value)
        if session is None:
            return JournalSnapshot(session_id, 0, 0)
        event_start = bisect_right(session.events, after_sequence, key=lambda entry: entry.sequence)
        transcript_start = bisect_right(
            session.transcript, after_sequence, key=lambda entry: entry.sequence
        )
        return _canonical_copy(
            JournalSnapshot(
                session_id,
                session.version,
                session.sequence,
                session.state,
                session.events[event_start:],
                session.transcript[transcript_start:],
                tuple(session.summaries.values()),
            )
        )

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionAcceptance | None:
        if type(session_id) is not SessionId or type(submission_id) is not SubmissionId:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        session_id = _canonical_copy(session_id)
        submission_id = _canonical_copy(submission_id)
        with self._lock:
            store = self._store

        session = store.sessions.get(session_id.value)
        accepted = session.submissions.get(submission_id.value) if session is not None else None
        if accepted is None:
            return None
        _, receipt = accepted
        assert receipt.submission is not None and receipt.state is not None
        return SubmissionAcceptance(
            _canonical_copy(receipt.submission), _canonical_copy(receipt.state), receipt.version
        )


_T = TypeVar("_T")


def _canonical_copy(value: _T) -> _T:
    """Rebuild record fields through trusted constructors, without instance copy protocols."""
    try:
        return _rebuild(value)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError, OverflowError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


def _rebuild(value: object) -> Any:
    if (
        value is None
        or type(value) is bool
        or type(value) is bytes
        or type(value) is float
        or type(value) is int
        or type(value) is str
        or type(value) is RunStatus
        or type(value) is RunErrorCode
        or type(value) is DomainEventType
    ):
        return value
    if type(value) is datetime and _is_canonical_timestamp(value):
        return value.replace()
    if type(value) is ArtifactRef:
        try:
            return snapshot_artifact_ref(value)
        except ValueError:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
    if type(value) is ExternalizedContent:
        return ExternalizedContent(value.inline_content, value.artifacts, value.summary)
    if type(value) is tuple:
        return tuple(_rebuild(item) for item in value)
    if type(value) is frozenset and all(type(item) is str for item in value):
        return value

    record_type = type(value)
    if not (
        any(record_type is allowed for allowed in (*_ID_TYPES, *_RECORD_TYPES))
        and is_dataclass(record_type)
    ):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)

    declared = fields(record_type)
    names = {member.name for member in declared}
    source = {
        name: member
        for name, member in dict.items(object.__getattribute__(value, "__dict__"))
        if type(name) is str and name in names
    }
    args = {member.name: _rebuild(source[member.name]) for member in declared if member.init}
    if record_type is DomainEvent:
        # Decode the source of the public payload and let DomainEvent validate it again.
        encoded = source["_payload_json"]
        if type(encoded) is not str:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return DomainEvent(**args, payload=json.loads(encoded))

    constructor: Any = record_type
    rebuilt = constructor(**args)
    if record_type is RunFailure:
        message = source["message"]
        if type(message) is not str or message != rebuilt.message:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
    return rebuilt


def _intent_value(value: object) -> _IntentValue:
    """Encode each rebuilt value with a disjoint tag before comparing acceptance intents."""
    if value is None:
        return ("none", None)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if type(value) is bytes:
        return ("bytes", value)
    if type(value) is datetime:
        return ("datetime", value.isoformat(timespec="microseconds"))
    if type(value) is ArtifactRef:
        try:
            ref = snapshot_artifact_ref(value)
        except ValueError:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        return (
            "artifact_ref",
            (
                ref.artifact_id,
                ref.media_type,
                ref.size_bytes,
                ref.sha256,
                ref.name,
                object.__getattribute__(ref, "_metadata_json"),
            ),
        )
    if type(value) is ExternalizedContent:
        rebuilt = ExternalizedContent(value.inline_content, value.artifacts, value.summary)
        inline = rebuilt.inline_content
        return (
            "externalized_content",
            (
                None
                if inline is None
                else (
                    inline.media_type,
                    inline.size_bytes,
                    object.__getattribute__(inline, "_encoded"),
                    object.__getattribute__(inline, "_is_binary"),
                ),
                _intent_value(rebuilt.artifacts),
                rebuilt.summary,
            ),
        )
    if type(value) is RunStatus or type(value) is RunErrorCode or type(value) is DomainEventType:
        return ("enum", (type(value).__name__, value.value))
    if (
        type(value) is SessionId
        or type(value) is RunId
        or type(value) is SubmissionId
        or type(value) is ProfileId
        or type(value) is EventId
    ):
        return ("id", (type(value).__name__, value.value))
    if type(value) is tuple:
        return ("tuple", tuple(_intent_value(item) for item in value))
    if type(value) is frozenset:
        return ("frozenset", tuple(sorted(value)))
    record_type = type(value)
    if any(record_type is allowed for allowed in _RECORD_TYPES) and is_dataclass(record_type):
        return (
            "record",
            (
                record_type.__name__,
                tuple(
                    _intent_value(object.__getattribute__(value, member.name))
                    for member in fields(record_type)
                ),
            ),
        )
    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)


def _validate_members(batch: JournalCommit, sequence: int, expected_version: int) -> None:
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
    if state.journal_version != expected_version + 1 or state.journal_sequence != next_sequence:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
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
        or not _valid_artifact_refs(summary.artifacts)
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


def _valid_artifact_refs(value: object) -> bool:
    if (
        type(value) is not tuple
        or len(value) > 16
        or any(type(ref) is not ArtifactRef for ref in value)
    ):
        return False
    try:
        for ref in value:
            snapshot_artifact_ref(ref)
    except ValueError:
        return False
    return True
