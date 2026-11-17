from dataclasses import replace

from src.core.errors import RunErrorCode
from src.core.events import DomainEvent
from src.core.ids import SessionId
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


def validate_commit_receipt(
    value: object,
    batch: JournalCommit,
    *,
    expected_version: int,
    previous_sequence: int,
) -> JournalSnapshot:
    """Validate an untrusted journal receipt and return a trusted projection."""
    try:
        if type(value) is not JournalSnapshot:
            raise ValueError
        session_id = object.__getattribute__(value, "session_id")
        version = object.__getattribute__(value, "version")
        sequence = object.__getattribute__(value, "sequence")
        state = object.__getattribute__(value, "state")
        events = object.__getattribute__(value, "events")
        transcript = object.__getattribute__(value, "transcript")
        summaries = object.__getattribute__(value, "summaries")
        submission = object.__getattribute__(value, "submission")

        expected_events = tuple(
            JournalEvent(index, event)
            for index, event in enumerate(batch.events, start=previous_sequence + 1)
        )
        transcript_start = previous_sequence + len(expected_events) + 1
        expected_transcript = tuple(
            replace(entry, sequence=index)
            for index, entry in enumerate(batch.transcript, start=transcript_start)
        )
        expected_summaries = (batch.summary,) if batch.summary is not None else ()
        expected_sequence = previous_sequence + len(expected_events) + len(expected_transcript)

        if (
            type(session_id) is not SessionId
            or session_id != batch.state.session_id
            or type(version) is not int
            or version != expected_version + 1
            or type(sequence) is not int
            or sequence != expected_sequence
            or type(state) is not RunState
            or state != batch.state
            or type(events) is not tuple
            or any(type(item) is not JournalEvent for item in events)
            or events != expected_events
            or type(transcript) is not tuple
            or any(type(item) is not TranscriptEntry for item in transcript)
            or transcript != expected_transcript
            or type(summaries) is not tuple
            or any(type(item) is not ConversationSummaryCheckpoint for item in summaries)
            or summaries != expected_summaries
            or submission is not None
            and type(submission) is not SubmissionRecord
            or submission != batch.submission
        ):
            raise ValueError
        if any(type(item.event) is not DomainEvent for item in events):
            raise ValueError
        return JournalSnapshot(
            batch.state.session_id,
            expected_version + 1,
            expected_sequence,
            batch.state,
            expected_events,
            expected_transcript,
            expected_summaries,
            batch.submission,
        )
    except Exception:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
