from src.core.errors import RunErrorCode
from src.core.journal import (
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalSnapshot,
)
from src.core.receipt_snapshot import (
    snapshot_artifact_attachment,
    snapshot_domain_event,
    snapshot_journal_snapshot,
    snapshot_run_state,
    snapshot_submission,
    snapshot_summary,
    snapshot_transcript,
)


def validate_commit_receipt(
    value: object,
    batch: JournalCommit,
    *,
    expected_version: int,
    previous_sequence: int,
) -> JournalSnapshot:
    """Validate an untrusted journal receipt and return a trusted projection."""
    try:
        receipt = snapshot_journal_snapshot(value)
        state = snapshot_run_state(batch.state)
        expected_events = tuple(
            JournalEvent(index, snapshot_domain_event(event))
            for index, event in enumerate(batch.events, start=previous_sequence + 1)
        )
        transcript_start = previous_sequence + len(expected_events) + 1
        expected_transcript = tuple(
            snapshot_transcript(entry, sequence=index)
            for index, entry in enumerate(batch.transcript, start=transcript_start)
        )
        expected_summaries = (snapshot_summary(batch.summary),) if batch.summary is not None else ()
        expected_submission = (
            snapshot_submission(batch.submission) if batch.submission is not None else None
        )
        expected_sequence = previous_sequence + len(expected_events) + len(expected_transcript)
        expected = JournalSnapshot(
            state.session_id,
            expected_version + 1,
            expected_sequence,
            state,
            expected_events,
            expected_transcript,
            expected_summaries,
            expected_submission,
            tuple(snapshot_artifact_attachment(fact) for fact in batch.artifact_facts),
        )
        if receipt != expected:
            raise ValueError
        return receipt
    except Exception:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
