import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from typing import cast

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.core.errors import RunErrorCode
from src.core.events import DomainEvent, JsonValue
from src.core.journal import JournalCommit, JournalError, JournalSnapshot
from tests.unit.core.test_journal_contract import acceptance


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
            assert submission == batch.submission
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
