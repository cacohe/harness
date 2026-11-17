import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
import src.adapters.persistence.sqlite as sqlite_module
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.adapters.persistence.sqlite_codec import decode, encode
from src.core.errors import RunErrorCode
from src.core.ids import EventId, SessionId
from src.core.journal import (
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalSnapshot,
    TranscriptEntry,
)
from src.core.state import RunStatus
from tests.unit.core.test_journal_contract import acceptance


def make_journal(tmp_path: Path) -> SQLiteRunJournal:
    return SQLiteRunJournal((tmp_path / "journal.sqlite3").resolve())


def test_sqlite_v3_round_trips_and_reopens(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("first")
    receipt = asyncio.run(journal.commit(batch, 0))
    reopened = make_journal(tmp_path)

    assert asyncio.run(reopened.load_session(batch.state.session_id)) == replace(
        receipt, submission=None
    )
    assert asyncio.run(reopened.load_head(batch.state.session_id)).state == batch.state
    found = asyncio.run(reopened.find_submission(batch.state.session_id, batch.state.submission_id))
    assert found is not None and (found.state, found.version) == (batch.state, 1)
    assert asyncio.run(reopened.commit(acceptance("first"), 0)) == receipt


def test_sqlite_commit_is_atomic_on_conflict(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("first")
    second = acceptance("second")
    asyncio.run(journal.commit(first, 0))
    before = asyncio.run(journal.load_session(first.state.session_id))
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.commit(second, 0))
    assert caught.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
    assert asyncio.run(journal.load_session(first.state.session_id)) == before


def test_two_connections_compete_for_same_session(tmp_path: Path) -> None:
    journals = (make_journal(tmp_path), make_journal(tmp_path))
    batches = (acceptance("first"), acceptance("second"))
    barrier = Barrier(2)

    def write(index: int):
        barrier.wait(5)
        try:
            return asyncio.run(journals[index].commit(batches[index], 0))
        except JournalError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(write, range(2)))
    assert sum(type(item) is JournalSnapshot for item in results) == 1
    assert sum(type(item) is JournalError for item in results) == 1


def test_two_connections_can_advance_different_sessions(tmp_path: Path) -> None:
    journals = (make_journal(tmp_path), make_journal(tmp_path))
    batches = (acceptance("first", session="one"), acceptance("second", session="two"))
    barrier = Barrier(2)

    def write(index: int):
        barrier.wait(5)
        return asyncio.run(journals[index].commit(batches[index], 0))

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = tuple(pool.map(write, range(2)))
    assert [(item.version, item.sequence) for item in receipts] == [(1, 2), (1, 2)]


def test_schema_is_exact_v3_and_has_required_indexes(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    with sqlite3.connect(journal.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {
            "sessions",
            "journal_entries",
            "runs",
            "events",
            "submissions",
            "transcript",
            "summary_checkpoints",
        }
        indexes = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert {
            "runs_active_by_session",
            "events_by_session_sequence",
            "transcript_by_session_sequence",
            "submissions_by_identity",
        } <= indexes


@pytest.mark.parametrize("version", [1, 2, 4])
def test_old_or_future_schema_is_rejected_without_mutation(tmp_path: Path, version: int) -> None:
    path = (tmp_path / "old.sqlite3").resolve()
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE legacy(value TEXT)")
        connection.execute(f"PRAGMA user_version={version}")
    with pytest.raises(ValueError):
        SQLiteRunJournal(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='legacy'").fetchone()


@pytest.mark.parametrize(
    "script",
    [
        "CREATE VIEW unknown_view AS SELECT 1;",
        "CREATE TABLE unknown(value INTEGER);"
        "CREATE TRIGGER unknown_trigger AFTER INSERT ON unknown BEGIN SELECT 1; END;",
        "CREATE TABLE unknown(value INTEGER);CREATE INDEX unknown_index ON unknown(value);",
    ],
)
def test_version_zero_unknown_objects_are_rejected_without_mutation(
    tmp_path: Path, script: str
) -> None:
    path = (tmp_path / "unknown.sqlite3").resolve()
    with sqlite3.connect(path) as connection:
        connection.executescript(script)
    before_bytes = path.read_bytes()
    with sqlite3.connect(path) as connection:
        before_objects = tuple(
            connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        )

    with pytest.raises(ValueError, match="schema v3"):
        SQLiteRunJournal(path)

    assert path.read_bytes() == before_bytes
    with sqlite3.connect(path) as connection:
        assert (
            tuple(
                connection.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_schema "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                )
            )
            == before_objects
        )


def test_load_missing_session_is_empty(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    assert asyncio.run(journal.load_session(SessionId("missing"))) == JournalSnapshot(
        SessionId("missing"), 0, 0
    )


def test_corrupt_codec_version_is_rejected_without_rewriting_storage(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("first")
    asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE sessions SET state_json = ? WHERE session_id = ?",
            ('{"codec":2,"value":null}', batch.state.session_id.value),
        )
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_head(batch.state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    with sqlite3.connect(journal.path) as connection:
        assert connection.execute("SELECT state_json FROM sessions").fetchone()[0] == (
            '{"codec":2,"value":null}'
        )


@pytest.mark.parametrize(
    "changed_ddl",
    [
        sqlite_module._DDL.replace("version INTEGER NOT NULL", "version INTEGER", 1),
        sqlite_module._DDL.replace("session_id TEXT PRIMARY KEY", "session_id TEXT", 1),
        sqlite_module._DDL.replace("ON DELETE CASCADE", "ON DELETE RESTRICT", 1),
        sqlite_module._DDL.replace(
            "runs_active_by_session ON runs(session_id, status)",
            "runs_active_by_session ON runs(status, session_id)",
            1,
        ),
        sqlite_module._DDL + "CREATE VIEW unexpected_view AS SELECT 1;",
        sqlite_module._DDL
        + "CREATE TRIGGER unexpected_trigger AFTER INSERT ON sessions BEGIN SELECT 1; END;",
    ],
)
def test_schema_definition_drift_is_rejected(tmp_path: Path, changed_ddl: str) -> None:
    path = (tmp_path / "drift.sqlite3").resolve()
    with sqlite3.connect(path) as connection:
        connection.executescript(changed_ddl)
        connection.execute("PRAGMA user_version=3")
    with pytest.raises(ValueError, match="schema v3"):
        SQLiteRunJournal(path)


def test_failed_initialization_leaves_no_partial_application_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = (tmp_path / "partial.sqlite3").resolve()
    monkeypatch.setattr(sqlite_module, "_DDL", sqlite_module._DDL + "INVALID SQL;")
    with pytest.raises(sqlite3.Error):
        SQLiteRunJournal(path)
    with sqlite3.connect(path) as connection:
        assert not {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("table", "column", "value", "operation"),
    [
        ("sessions", "version", 99, "head"),
        ("sessions", "sequence", 99, "session"),
        ("runs", "status", "failed", "head"),
        ("runs", "state_version", 99, "head"),
        ("events", "sequence", 99, "session"),
        ("events", "event_id", "wrong", "session"),
        ("events", "run_id", "wrong", "session"),
        ("transcript", "run_id", "wrong", "session"),
        ("transcript", "content", "wrong", "session"),
        ("summary_checkpoints", "covered_from_sequence", 2, "session"),
        ("summary_checkpoints", "profile_id", "wrong", "session"),
        ("submissions", "digest", "wrong", "submission"),
        ("submissions", "acceptance_version", 99, "submission"),
    ],
)
def test_redundant_index_columns_must_match_payload(
    tmp_path: Path, table: str, column: str, value: object, operation: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("first")
    asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute(f"UPDATE {table} SET {column} = ?", (value,))
    with pytest.raises(JournalError) as caught:
        if operation == "head":
            asyncio.run(journal.load_head(batch.state.session_id))
        elif operation == "submission":
            asyncio.run(journal.find_submission(batch.state.session_id, batch.state.submission_id))
        else:
            asyncio.run(journal.load_session(batch.state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_event_payload_identity_must_match_index_columns(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("first")
    receipt = asyncio.run(journal.commit(batch, 0))
    source = receipt.events[0]
    forged_event = type(source.event)(
        EventId("forged"),
        source.event.type,
        source.event.run_id,
        source.event.session_id,
        source.event.timestamp,
        source.event.payload,
    )
    forged = JournalEvent(source.sequence, forged_event)
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE events SET event_json = ?", (encode(forged),))
    with pytest.raises(JournalError):
        asyncio.run(journal.load_session(batch.state.session_id))


def test_state_journal_metadata_must_match_session_and_run_rows(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("metadata")
    asyncio.run(journal.commit(batch, 0))
    forged = replace(batch.state, journal_version=99)
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE sessions SET state_json = ?", (encode(forged),))
        connection.execute("UPDATE runs SET state_json = ?", (encode(forged),))

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_head(batch.state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize("corruption", ["gap", "cross_table_reorder"])
def test_shared_sequence_must_be_dense_and_globally_ordered(
    tmp_path: Path, corruption: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("sequence")
    receipt = asyncio.run(journal.commit(batch, 0))
    event = receipt.events[0]
    transcript = receipt.transcript[0]
    with sqlite3.connect(journal.path) as connection:
        if corruption == "gap":
            forged_state = replace(batch.state, journal_sequence=3)
            connection.execute(
                "UPDATE sessions SET sequence=3, state_json=?", (encode(forged_state),)
            )
            connection.execute("UPDATE runs SET state_json=?", (encode(forged_state),))
            connection.execute(
                "UPDATE transcript SET sequence=3, entry_json=?",
                (encode(replace(transcript, sequence=3)),),
            )
        else:
            connection.execute(
                "UPDATE events SET sequence=2, event_json=?",
                (encode(JournalEvent(2, event.event)),),
            )
            connection.execute(
                "UPDATE transcript SET sequence=1, entry_json=?",
                (encode(replace(transcript, sequence=1)),),
            )

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_session(batch.state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_shared_sequence_preserves_kind_for_non_acceptance_commits(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("sequence-kind")
    asyncio.run(journal.commit(first, 0))
    state = replace(
        first.state,
        version=2,
        status=RunStatus.COMPLETED,
        final_text="done",
        journal_version=2,
        journal_sequence=4,
    )
    event = type(first.events[0])(
        EventId("later-event"),
        first.events[0].type,
        state.run_id,
        state.session_id,
        first.events[0].timestamp,
        {},
    )
    entry = TranscriptEntry(
        state.session_id,
        state.run_id,
        state.submission_id,
        "assistant",
        "done",
    )
    receipt = asyncio.run(
        journal.commit(JournalCommit(state, events=(event,), transcript=(entry,)), 1)
    )
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE events SET sequence=4, event_json=? WHERE event_id='later-event'",
            (encode(JournalEvent(4, event)),),
        )
        connection.execute(
            "UPDATE transcript SET sequence=3, entry_json=? WHERE sequence=4",
            (encode(replace(receipt.transcript[0], sequence=3)),),
        )

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_session(state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize(
    ("member", "operation"),
    [
        ("events", "find"),
        ("transcript", "find"),
        ("summary", "find"),
        ("events", "replay"),
        ("transcript", "replay"),
        ("summary", "replay"),
    ],
)
def test_submission_intent_must_fully_match_original_receipt(
    tmp_path: Path, member: str, operation: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("intent")
    asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        encoded = connection.execute("SELECT intent_json FROM submissions").fetchone()[0]
        intent = decode(encoded, JournalCommit)
        changed = replace(intent, **{member: () if member != "summary" else None})
        connection.execute("UPDATE submissions SET intent_json=?", (encode(changed),))

    with pytest.raises(JournalError) as caught:
        if operation == "find":
            asyncio.run(journal.find_submission(batch.state.session_id, batch.state.submission_id))
        else:
            asyncio.run(journal.commit(batch, 0))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_corrupt_existing_summary_key_blocks_commit_without_writes(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("summary-row")
    asyncio.run(journal.commit(batch, 0))
    next_state = replace(
        batch.state,
        version=2,
        journal_version=2,
        journal_sequence=2,
    )
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE summary_checkpoints SET covered_from_sequence=2")
        before = tuple(connection.execute("SELECT version, sequence, state_json FROM sessions"))

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.commit(JournalCommit(next_state, summary=batch.summary), 1))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    with sqlite3.connect(journal.path) as connection:
        assert (
            tuple(connection.execute("SELECT version, sequence, state_json FROM sessions"))
            == before
        )


@pytest.mark.parametrize(
    "malformed",
    [
        "[" * 2_000 + "0" + "]" * 2_000,
        '{"codec":1,"value":{"$":"tuple","items":['
        + ",".join("null" for _ in range(4_097))
        + "]}}",
        '{"codec":1,"value":NaN}',
        '{"codec":1,"codec":1,"value":null}',
        '{"codec":1,"value":{"$":"unknown"}}',
        '{"codec":1,"value":' + '{"$":"tuple","items":[' * 70 + "null" + "]}" * 70 + "}",
    ],
)
def test_codec_structural_attacks_map_to_stable_journal_failure(
    tmp_path: Path, malformed: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("first")
    asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE sessions SET state_json = ?", (malformed,))
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_head(batch.state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
