import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from typing import cast

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.core.errors import RunErrorCode
from src.core.events import DomainEvent, JsonValue
from src.core.ids import EventId, SessionId
from src.core.journal import JournalCommit, JournalError, JournalSnapshot
from tests.unit.core.test_journal_contract import acceptance


def test_instance_copy_hook_is_ignored_for_exact_journal_records() -> None:
    def unexpected_copy(_memo: object) -> object:
        pytest.fail("journal must not call an instance copy hook")

    journal = InMemoryRunJournal()
    batch = acceptance("first")
    assert batch.submission is not None and batch.summary is not None
    for record in (batch, batch.state, batch.events[0], batch.submission, batch.summary):
        object.__setattr__(record, "__deepcopy__", unexpected_copy)

    receipt = asyncio.run(journal.commit(batch, expected_version=0))

    assert receipt.state == batch.state and receipt.state is not batch.state
    assert receipt.events[0].event == batch.events[0]
    assert receipt.events[0].event is not batch.events[0]
    assert receipt.submission == batch.submission and receipt.submission is not batch.submission
    assert receipt.summaries == (batch.summary,) and receipt.summaries[0] is not batch.summary
    assert asyncio.run(journal.commit(acceptance("first"), expected_version=0)) == receipt


def test_event_storage_and_index_use_the_same_canonical_copy() -> None:
    journal = InMemoryRunJournal()
    batch = acceptance("first")
    event = batch.events[0]
    alternate = DomainEvent(
        EventId("alternate"), event.type, event.run_id, event.session_id, event.timestamp, {}
    )
    object.__setattr__(event, "__deepcopy__", lambda memo: alternate)

    receipt = asyncio.run(journal.commit(batch, expected_version=0))
    assert receipt.events[0].event == event
    assert receipt.events[0].event.event_id is not event.event_id
    object.__setattr__(event.event_id, "value", "changed-input-id")
    object.__setattr__(receipt.events[0].event.event_id, "value", "changed-output-id")

    stored = asyncio.run(journal.load_session(batch.state.session_id))
    assert stored.events[0].event.event_id == EventId("event-first")
    assert stored.events[0].event.timestamp.tzinfo is event.timestamp.tzinfo
    replay = asyncio.run(journal.commit(acceptance("first"), expected_version=0))
    assert replay == replace(stored, submission=batch.submission)
    repeated = acceptance("second", event_id="event-first")
    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(repeated, expected_version=1))
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.load_session(batch.state.session_id)) == stored
    available = acceptance("second", event_id="alternate")
    assert asyncio.run(journal.commit(available, expected_version=1)).version == 2


def test_unhashable_checkpoint_identity_rejects_every_commit_member() -> None:
    class UnhashableSession(SessionId):
        def __eq__(self, other: object) -> bool:
            return isinstance(other, SessionId) and self.value == other.value

        __hash__ = None

    journal = InMemoryRunJournal()
    first = acceptance("first")
    pending = acceptance("next")
    session_id = first.state.session_id
    assert pending.summary is not None
    asyncio.run(journal.commit(first, expected_version=0))
    before = asyncio.run(journal.load_session(session_id))
    malformed = replace(pending.summary)
    object.__setattr__(malformed, "session_id", UnhashableSession(session_id.value))

    with pytest.raises((TypeError, JournalError)) as error:
        asyncio.run(journal.commit(replace(pending, summary=malformed), expected_version=1))

    assert asyncio.run(journal.load_session(session_id)) == before
    assert asyncio.run(journal.find_submission(session_id, pending.state.submission_id)) is None
    assert isinstance(error.value, JournalError)
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    committed = asyncio.run(journal.commit(pending, expected_version=1))
    assert (committed.version, committed.sequence) == (2, 4)
    assert committed.events[0].event.event_id == pending.events[0].event_id


@pytest.mark.parametrize("session", ["session", "another"])
def test_duplicate_event_id_in_later_commit_rejects_all_members(session: str) -> None:
    journal = InMemoryRunJournal()
    first = acceptance("first", event_id="shared")
    second = acceptance("second", session=session, event_id="shared")
    asyncio.run(journal.commit(first, expected_version=0))
    before = asyncio.run(journal.load_session(second.state.session_id))

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(second, expected_version=before.version))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.load_session(second.state.session_id)) == before
    assert (
        asyncio.run(journal.find_submission(second.state.session_id, second.state.submission_id))
        is None
    )


def test_duplicate_event_id_within_batch_can_be_retried() -> None:
    journal = InMemoryRunJournal()
    valid = acceptance("first")
    batch = replace(valid, events=valid.events * 2)

    with pytest.raises(JournalError) as error:
        asyncio.run(journal.commit(batch, expected_version=0))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.load_session(valid.state.session_id)).version == 0
    assert asyncio.run(journal.commit(valid, expected_version=0)).version == 1


def test_competing_threads_commit_exactly_one_complete_batch() -> None:
    journal = InMemoryRunJournal()
    contenders = (acceptance("first"), acceptance("second"))
    ready = Barrier(2)

    def write(batch: JournalCommit) -> JournalSnapshot | JournalError:
        ready.wait(timeout=5)
        try:
            return asyncio.run(journal.commit(batch, expected_version=0))
        except JournalError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(write, contenders))

    assert sum(isinstance(result, JournalSnapshot) for result in results) == 1
    for batch, result in zip(contenders, results, strict=True):
        session = batch.state.session_id
        submission = asyncio.run(journal.find_submission(session, batch.state.submission_id))
        snapshot = asyncio.run(journal.load_session(session))
        assert (snapshot.version, snapshot.sequence) == (1, 2)
        if isinstance(result, JournalError):
            assert result.failure.code is RunErrorCode.JOURNAL_CONFLICT
            assert submission is None
            assert batch.state != snapshot.state
            assert batch.events[0] != snapshot.events[0].event
            assert batch.transcript[0].content != snapshot.transcript[0].content
            assert batch.summary not in snapshot.summaries
        else:
            assert submission is not None and submission.submission == batch.submission
            assert snapshot.state == batch.state
            assert [record.event for record in snapshot.events] == list(batch.events)
            assert [record.content for record in snapshot.transcript] == [
                batch.transcript[0].content
            ]
            assert snapshot.summaries == (batch.summary,)


def test_nested_json_is_isolated_from_writes_and_snapshot_reads() -> None:
    journal = InMemoryRunJournal()
    batch = acceptance("first")
    template = batch.events[0]
    messages: list[JsonValue] = ["before"]
    event = DomainEvent(
        template.event_id,
        template.type,
        template.run_id,
        template.session_id,
        template.timestamp,
        {"nested": {"messages": messages}},
    )
    receipt = asyncio.run(journal.commit(replace(batch, events=(event,)), expected_version=0))
    messages.append("caller mutation")
    nested = cast(dict[str, JsonValue], receipt.events[0].event.payload["nested"])
    cast(list[JsonValue], nested["messages"]).append("reader mutation")

    later = asyncio.run(journal.load_session(batch.state.session_id))
    assert later.events[0].event.payload == {"nested": {"messages": ["before"]}}
    assert isinstance(later.events, tuple)
    assert isinstance(later.transcript, tuple)
    assert isinstance(later.summaries, tuple)
    assert receipt.state is not batch.state
    assert later.state is not receipt.state
    assert later.events[0].event is not event


@pytest.mark.parametrize("cursor", [-1, False, 1.5])
def test_invalid_load_cursor_has_stable_failure(cursor: object) -> None:
    journal = InMemoryRunJournal()
    session_id = acceptance("first").state.session_id
    with pytest.raises(JournalError) as error:
        asyncio.run(journal.load_session(session_id, cast(int, cursor)))
    failure = error.value.failure
    assert failure.code is RunErrorCode.INVARIANT_VIOLATION
