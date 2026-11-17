import asyncio
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone, tzinfo

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.core.commands import RunLimits
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    RunJournal,
    SubmissionRecord,
    TranscriptEntry,
)
from src.core.state import RunState, RunStatus

NOW = datetime(2026, 11, 17, tzinfo=timezone.utc)
LIMITS = RunLimits(5, 30, 1024)


@pytest.fixture
def journal() -> RunJournal:
    return InMemoryRunJournal()


def acceptance(
    name: str, *, session: str = "session", event_id: str | None = None
) -> JournalCommit:
    state = RunState(
        RunId(f"run-{name}"),
        SessionId(session),
        SubmissionId(f"submit-{name}"),
        ProfileId("default"),
        LIMITS,
        version=1,
        status=RunStatus.RUNNING,
    )
    event = DomainEvent(
        EventId(event_id or f"event-{name}"),
        DomainEventType.RUN_STARTED,
        state.run_id,
        state.session_id,
        NOW,
        {"nested": {"messages": ["original"]}},
    )
    return JournalCommit(
        state=state,
        events=(event,),
        submission=SubmissionRecord(state.session_id, state.run_id, state.submission_id, name),
        transcript=(
            TranscriptEntry(state.session_id, state.run_id, state.submission_id, "user", name),
        ),
        summary=ConversationSummaryCheckpoint(
            state.session_id, 2, "v1", state.profile_id, f"summary-{name}"
        ),
    )


def test_commit_schema_ignores_instance_field_metadata(journal: RunJournal) -> None:
    batch = acceptance("first")
    reduced_fields = {
        member.name: member
        for member in fields(JournalCommit)
        if member.name not in ("events", "summary")
    }
    object.__setattr__(batch, "__dataclass_fields__", reduced_fields)

    receipt = asyncio.run(journal.commit(batch, expected_version=0))
    persisted = asyncio.run(journal.load_session(batch.state.session_id))

    assert (receipt.version, receipt.sequence) == (1, 2)
    assert persisted.events[0].event == batch.events[0]
    assert persisted.summaries == (batch.summary,)
    assert asyncio.run(journal.commit(acceptance("first"), expected_version=0)) == receipt
    duplicate = acceptance("second", event_id="event-first")
    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(duplicate, expected_version=1))
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.load_session(batch.state.session_id)) == persisted


def test_nested_record_schema_does_not_call_instance_metadata(journal: RunJournal) -> None:
    class UnexpectedFields:
        def values(self):
            raise RuntimeError("journal must use the trusted record class schema")

    batch = acceptance("first")
    object.__setattr__(batch.events[0], "__dataclass_fields__", UnexpectedFields())

    receipt = asyncio.run(journal.commit(batch, expected_version=0))

    assert (receipt.version, receipt.sequence) == (1, 2)
    assert receipt.events[0].event == batch.events[0]
    assert receipt.summaries == (batch.summary,)
    assert receipt.submission == batch.submission


def test_session_sequence_orders_events_before_transcript(journal: RunJournal) -> None:
    batch = acceptance("first")
    state = batch.state
    first = asyncio.run(journal.commit(batch, expected_version=0))
    state_only = asyncio.run(
        journal.commit(JournalCommit(replace(state, version=4)), expected_version=1)
    )
    end_event = DomainEvent(
        EventId("end"), DomainEventType.RUN_COMPLETED, state.run_id, state.session_id, NOW, {}
    )
    terminal = replace(state, version=6, status=RunStatus.COMPLETED, final_text="done")
    second = asyncio.run(
        journal.commit(
            JournalCommit(
                terminal,
                events=(end_event,),
                transcript=(
                    TranscriptEntry(
                        state.session_id, state.run_id, state.submission_id, "assistant", "done"
                    ),
                ),
            ),
            expected_version=2,
        )
    )

    assert (first.version, first.sequence) == (1, 2)
    assert (state_only.version, state_only.sequence) == (2, 2)
    assert (second.version, second.sequence) == (3, 4)
    assert [entry.sequence for entry in second.events] == [3]
    assert [entry.sequence for entry in second.transcript] == [4]

    all_entries = asyncio.run(journal.load_session(state.session_id))
    assert all_entries.state == terminal
    assert [entry.sequence for entry in all_entries.events] == [1, 3]
    assert [entry.sequence for entry in all_entries.transcript] == [2, 4]
    suffix = asyncio.run(journal.load_session(state.session_id, after_sequence=3))
    assert (suffix.version, suffix.sequence, suffix.state) == (3, 4, terminal)
    assert suffix.events == ()
    assert [entry.sequence for entry in suffix.transcript] == [4]
    assert suffix.summaries == all_entries.summaries
    beyond_head = asyncio.run(journal.load_session(state.session_id, after_sequence=10))
    assert beyond_head.events == beyond_head.transcript == ()

    independent = asyncio.run(journal.commit(acceptance("other", session="other"), 0))
    assert (independent.version, independent.sequence) == (1, 2)


def test_expected_version_conflict_leaves_prior_snapshot_intact(journal: RunJournal) -> None:
    initial = acceptance("first")
    pending = acceptance("pending")
    asyncio.run(journal.commit(initial, expected_version=0))
    before = asyncio.run(journal.load_session(initial.state.session_id))

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(pending, expected_version=0))

    assert error.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
    assert asyncio.run(journal.load_session(initial.state.session_id)) == before
    assert (
        asyncio.run(journal.find_submission(pending.state.session_id, pending.state.submission_id))
        is None
    )
    asyncio.run(journal.commit(pending, expected_version=1))


@pytest.mark.parametrize(
    "invalid_field",
    [
        "event_session",
        "event_run",
        "event_timestamp",
        "submission_session",
        "submission_run",
        "submission_id",
        "submission_digest",
        "transcript_session",
        "transcript_run",
        "transcript_submission",
        "transcript_sequence",
        "summary_session",
        "summary_policy",
        "summary_ahead",
        "summary_covered_range",
        "state",
    ],
)
def test_invalid_commit_member_leaves_the_session_empty(
    journal: RunJournal, invalid_field: str
) -> None:
    valid = acceptance("valid")
    state = valid.state
    event = valid.events[0]
    transcript = valid.transcript[0]
    submission = valid.submission
    summary = valid.summary
    assert submission is not None and summary is not None
    malformed_event = {
        "event_session": (state.run_id, SessionId("wrong"), NOW),
        "event_run": (RunId("wrong"), state.session_id, NOW),
        "event_timestamp": (state.run_id, state.session_id, "invalid"),
    }
    malformed = {
        "submission_session": {"submission": replace(submission, session_id=SessionId("wrong"))},
        "submission_run": {"submission": replace(submission, run_id=RunId("wrong"))},
        "submission_id": {"submission": replace(submission, submission_id=SubmissionId("wrong"))},
        "submission_digest": {"submission": replace(submission, digest="")},
        "transcript_session": {"transcript": (replace(transcript, session_id=SessionId("wrong")),)},
        "transcript_run": {"transcript": (replace(transcript, run_id=RunId("wrong")),)},
        "transcript_submission": {
            "transcript": (replace(transcript, submission_id=SubmissionId("wrong")),)
        },
        "transcript_sequence": {"transcript": (replace(transcript, sequence=1),)},
        "summary_session": {"summary": replace(summary, session_id=SessionId("wrong"))},
        "summary_policy": {"summary": replace(summary, policy_version="")},
        "summary_ahead": {"summary": replace(summary, covered_through_sequence=3)},
        "summary_covered_range": {"summary": replace(summary, covered_from_sequence=3)},
        "state": {"state": None},
    }
    if invalid_field in malformed_event:
        run_id, session_id, timestamp = malformed_event[invalid_field]
        bad_event = DomainEvent(event.event_id, event.type, run_id, session_id, NOW, {})
        object.__setattr__(bad_event, "timestamp", timestamp)
        batch = replace(valid, events=(bad_event,))
    else:
        batch = replace(valid, **malformed[invalid_field])

    before = asyncio.run(journal.load_session(state.session_id))
    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(batch, expected_version=0))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.load_session(state.session_id)) == before
    assert asyncio.run(journal.find_submission(state.session_id, state.submission_id)) is None
    assert asyncio.run(journal.commit(valid, expected_version=0)).version == 1


def test_submission_retry_returns_original_commit_receipt(journal: RunJournal) -> None:
    original = acceptance("first")
    session = original.state.session_id
    receipt = asyncio.run(journal.commit(original, expected_version=0))
    asyncio.run(journal.commit(acceptance("next"), expected_version=1))

    retry = asyncio.run(journal.commit(acceptance("first"), expected_version=0))

    assert retry == receipt
    assert retry.submission == original.submission
    assert asyncio.run(journal.find_submission(session, original.state.submission_id)) == (
        original.submission
    )
    assert asyncio.run(journal.load_session(session)).version == 2


def _event_with_timestamp(event: DomainEvent, timestamp: datetime) -> DomainEvent:
    return DomainEvent(
        event.event_id, event.type, event.run_id, event.session_id, timestamp, event.payload
    )


@pytest.mark.parametrize(
    "change", ["digest", "state", "events", "timestamp", "transcript", "summary"]
)
def test_reused_submission_id_with_changed_intent_conflicts(
    journal: RunJournal, change: str
) -> None:
    original = acceptance("first")
    state = original.state
    assert original.submission is not None
    asyncio.run(journal.commit(original, expected_version=0))
    before = asyncio.run(journal.load_session(state.session_id))
    updates = {
        "digest": {"submission": replace(original.submission, digest="changed")},
        "state": {"state": replace(state, version=2)},
        "events": {"events": ()},
        "timestamp": {
            "events": (_event_with_timestamp(original.events[0], NOW + timedelta(seconds=1)),)
        },
        "transcript": {"transcript": ()},
        "summary": {"summary": None},
    }

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(replace(original, **updates[change]), expected_version=1))

    assert error.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
    assert asyncio.run(journal.load_session(state.session_id)) == before


class MutableFoldTimezone(tzinfo):
    offset = -4

    def utcoffset(self, dt):
        return timedelta(hours=self.offset - dt.fold)

    def __deepcopy__(self, memo):
        return self


@pytest.mark.parametrize("fold", [0, 1])
def test_acceptance_replay_rejects_untrusted_timezone(journal: RunJournal, fold: int) -> None:
    original = acceptance("first")
    receipt = asyncio.run(journal.commit(original, expected_version=0))
    template = original.events[0]
    invalid = _event_with_timestamp(template, NOW)
    local_timestamp = datetime(2026, 11, 1, 1, 30, tzinfo=MutableFoldTimezone(), fold=fold)
    object.__setattr__(invalid, "timestamp", local_timestamp)

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(replace(original, events=(invalid,)), expected_version=0))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.commit(acceptance("first"), expected_version=0)) == receipt


def test_same_submission_id_is_independent_between_sessions(journal: RunJournal) -> None:
    first = acceptance("same", session="first", event_id="event-1")
    second = acceptance("same", session="second", event_id="event-2")
    asyncio.run(journal.commit(first, expected_version=0))
    asyncio.run(journal.commit(second, expected_version=0))

    for batch in (first, second):
        assert (
            asyncio.run(journal.find_submission(batch.state.session_id, batch.state.submission_id))
            == batch.submission
        )


@pytest.mark.parametrize("message", [False, ("bool", 0)])
def test_replay_rejects_tampered_failure_message(journal: RunJournal, message: object) -> None:
    batch = acceptance("first")
    failure = RunFailure(RunErrorCode.MODEL_FAILED)
    original = replace(batch, state=replace(batch.state, status=RunStatus.FAILED, failure=failure))
    receipt = asyncio.run(journal.commit(original, expected_version=0))
    corrupt = RunFailure(RunErrorCode.MODEL_FAILED)
    object.__setattr__(corrupt, "message", message)
    invalid = replace(original, state=replace(original.state, failure=corrupt))

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(invalid, expected_version=1))
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.commit(original, expected_version=0)) == receipt


@pytest.mark.parametrize("suffix", ["Z", "+00:00"])
def test_utc_iso_timestamp_round_trips(journal: RunJournal, suffix: str) -> None:
    batch = acceptance("first")
    timestamp = datetime.fromisoformat("2026-11-17T00:00:00" + suffix)
    event = _event_with_timestamp(batch.events[0], timestamp)

    receipt = asyncio.run(journal.commit(replace(batch, events=(event,)), expected_version=0))

    persisted = receipt.events[0].event.timestamp
    assert persisted == NOW and persisted.tzinfo is timezone.utc
    assert persisted is not timestamp


def test_lookup_rejects_malformed_identifiers(journal: RunJournal) -> None:
    session_id = SessionId("first")
    submission_id = SubmissionId("first")
    object.__delattr__(session_id, "value")
    with pytest.raises(JournalError) as error:
        asyncio.run(journal.load_session(session_id))
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION

    object.__delattr__(submission_id, "value")
    with pytest.raises(JournalError) as error:
        asyncio.run(journal.find_submission(SessionId("first"), submission_id))
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_summary_checkpoints_use_all_key_components(journal: RunJournal) -> None:
    initial = acceptance("first")
    state = initial.state
    first = initial.summary
    assert first is not None
    asyncio.run(journal.commit(initial, expected_version=0))
    variations = (
        replace(first, covered_through_sequence=1, summary="shorter"),
        replace(first, policy_version="v2", summary="policy"),
        replace(first, profile_id=ProfileId("summarizer"), summary="profile"),
    )
    for version, checkpoint in enumerate(variations, start=1):
        asyncio.run(journal.commit(JournalCommit(state, summary=checkpoint), version))
    other = acceptance("other", session="other")
    asyncio.run(journal.commit(other, expected_version=0))

    summaries = asyncio.run(journal.load_session(state.session_id)).summaries
    assert set(summaries) == {first, *variations}
    assert first.key == (state.session_id, 2, "v1", state.profile_id)
    assert asyncio.run(journal.load_session(other.state.session_id)).summaries == (other.summary,)


def test_same_run_cannot_regress_its_transition_version(journal: RunJournal) -> None:
    original = acceptance("first")
    state = original.state
    asyncio.run(journal.commit(original, expected_version=0))
    asyncio.run(journal.commit(JournalCommit(replace(state, version=4)), expected_version=1))
    before = asyncio.run(journal.load_session(state.session_id))

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(JournalCommit(replace(state, version=3)), expected_version=2))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    after = asyncio.run(journal.load_session(state.session_id))
    assert after == before
