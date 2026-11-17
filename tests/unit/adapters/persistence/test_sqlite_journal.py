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
from src.core.state import RunState, RunStatus
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
            "  previous_hash TEXT NOT NULL,\n  record_hash TEXT NOT NULL,\n", ""
        ),
        sqlite_module._DDL.replace(
            "  state_json TEXT NOT NULL,\n  ledger_hash TEXT NOT NULL\n",
            "  state_json TEXT NOT NULL\n",
            1,
        ),
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


def _append_second_pair(journal: SQLiteRunJournal, first: JournalCommit) -> JournalSnapshot:
    state = replace(
        first.state,
        version=2,
        journal_version=2,
        journal_sequence=4,
    )
    event = type(first.events[0])(
        EventId("second-pair-event"),
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
        "second pair",
    )
    return asyncio.run(
        journal.commit(JournalCommit(state, events=(event,), transcript=(entry,)), 1)
    )


def _raw_session_rows(path: Path, session_id: SessionId) -> tuple[tuple[str, tuple], ...]:
    result = []
    with sqlite3.connect(path) as connection:
        for table, order in (
            ("sessions", "session_id"),
            ("runs", "run_id"),
            ("journal_entries", "sequence"),
            ("events", "sequence"),
            ("transcript", "sequence"),
            ("summary_checkpoints", "covered_through_sequence, policy_version, profile_id"),
            ("submissions", "submission_id"),
        ):
            rows = tuple(
                connection.execute(
                    f"SELECT * FROM {table} WHERE session_id=? ORDER BY {order}",
                    (session_id.value,),
                )
            )
            result.append((table, rows))
    return tuple(result)


@pytest.mark.parametrize("with_records", [False, True], ids=["state-only", "with-records"])
@pytest.mark.parametrize(
    "corruption", ["synchronized-reorder", "synchronized-content", "synchronized-event"]
)
def test_independent_chain_rejects_self_consistent_history_rewrites(
    tmp_path: Path, corruption: str, with_records: bool
) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("chain")
    asyncio.run(journal.commit(first, 0))
    receipt = _append_second_pair(journal, first)
    attempted = _new_commit_after(receipt.state, version=2, sequence=4, with_records=with_records)
    with sqlite3.connect(journal.path) as connection:
        if corruption == "synchronized-reorder":
            connection.execute(
                "UPDATE events SET sequence=4, event_json=? WHERE sequence=3",
                (encode(replace(receipt.events[0], sequence=4)),),
            )
            connection.execute(
                "UPDATE transcript SET sequence=3, entry_json=? WHERE sequence=4",
                (encode(replace(receipt.transcript[0], sequence=3)),),
            )
            connection.execute("UPDATE journal_entries SET kind='transcript' WHERE sequence=3")
            connection.execute("UPDATE journal_entries SET kind='event' WHERE sequence=4")
        elif corruption == "synchronized-content":
            changed = replace(receipt.transcript[0], content="rewritten history")
            connection.execute(
                "UPDATE transcript SET content=?, entry_json=? WHERE sequence=4",
                (changed.content, encode(changed)),
            )
        else:
            event = receipt.events[0].event
            changed_event = type(event)(
                EventId("rewritten-event"),
                event.type,
                event.run_id,
                event.session_id,
                event.timestamp,
                {"fact": "rewritten"},
            )
            connection.execute(
                "UPDATE events SET event_id=?, event_json=? WHERE sequence=3",
                (changed_event.event_id.value, encode(JournalEvent(3, changed_event))),
            )
    before = _raw_session_rows(journal.path, first.state.session_id)
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.commit(attempted, 2))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert _raw_session_rows(journal.path, first.state.session_id) == before
    operations = [
        lambda: journal.load_session(first.state.session_id, after_sequence=4),
        lambda: journal.find_submission(first.state.session_id, first.state.submission_id),
        lambda: journal.commit(first, 0),
    ]
    if corruption != "synchronized-event":
        operations.append(lambda: journal.load_head(first.state.session_id))
    for operation in operations:
        with pytest.raises(JournalError) as caught:
            asyncio.run(operation())
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert _raw_session_rows(journal.path, first.state.session_id) == before


@pytest.mark.parametrize("with_records", [False, True], ids=["state-only", "with-records"])
@pytest.mark.parametrize(
    "corruption",
    ["root", "previous", "record", "uppercase", "short", "truncate", "insert"],
)
def test_chain_hash_corruption_rejects_reads_and_writes(
    tmp_path: Path, corruption: str, with_records: bool
) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("chain-fields")
    asyncio.run(journal.commit(first, 0))
    receipt = _append_second_pair(journal, first)
    with sqlite3.connect(journal.path) as connection:
        sequence = 4
        if corruption == "root":
            connection.execute("UPDATE sessions SET ledger_hash=?", ("0" * 64,))
        elif corruption in ("previous", "record", "uppercase", "short"):
            column = "previous_hash" if corruption == "previous" else "record_hash"
            value = "0" * 64
            if corruption == "uppercase":
                value = "A" * 64
            elif corruption == "short":
                value = "0"
            connection.execute(f"UPDATE journal_entries SET {column}=? WHERE sequence=4", (value,))
        else:
            if corruption == "truncate":
                connection.execute("DELETE FROM events WHERE sequence=3")
                connection.execute("DELETE FROM transcript WHERE sequence=4")
                connection.execute("DELETE FROM journal_entries WHERE sequence>2")
                sequence = 2
            else:
                extra = replace(receipt.transcript[0], sequence=5)
                connection.execute(
                    "INSERT INTO transcript(session_id, sequence, run_id, submission_id, role, "
                    "content, entry_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        extra.session_id.value,
                        5,
                        extra.run_id.value,
                        extra.submission_id.value,
                        extra.role,
                        extra.content,
                        encode(extra),
                    ),
                )
                root = connection.execute("SELECT ledger_hash FROM sessions").fetchone()[0]
                connection.execute(
                    "INSERT INTO journal_entries(session_id, sequence, kind, previous_hash, "
                    "record_hash) VALUES (?, 5, 'transcript', ?, ?)",
                    (first.state.session_id.value, root, "0" * 64),
                )
                sequence = 5
            # Also synchronize the head and Run counters: only the old chain/root remain.
            state_json = encode(replace(receipt.state, journal_sequence=sequence))
            connection.execute(
                "UPDATE sessions SET sequence=?, state_json=?", (sequence, state_json)
            )
            connection.execute("UPDATE runs SET state_json=?", (state_json,))
    before = _raw_session_rows(journal.path, first.state.session_id)
    attempted = _new_commit_after(
        receipt.state, version=2, sequence=sequence, with_records=with_records
    )
    for operation in (
        lambda: journal.commit(attempted, 2),
        lambda: journal.load_session(first.state.session_id, after_sequence=99),
        lambda: journal.load_head(first.state.session_id),
        lambda: journal.find_submission(first.state.session_id, first.state.submission_id),
        lambda: journal.commit(first, 0),
    ):
        with pytest.raises(JournalError) as caught:
            asyncio.run(operation())
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert _raw_session_rows(journal.path, first.state.session_id) == before


@pytest.mark.parametrize("empty", [False, True])
def test_chain_cannot_be_transplanted_between_sessions(tmp_path: Path, empty: bool) -> None:
    journal = make_journal(tmp_path)
    batches = (acceptance("a", session="a"), acceptance("b", session="b"))
    if empty:
        batches = tuple(JournalCommit(replace(b.state, journal_sequence=0)) for b in batches)
    for batch in batches:
        asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE sessions SET ledger_hash=(SELECT ledger_hash FROM sessions WHERE session_id='a') "
            "WHERE session_id='b'"
        )
        connection.execute(
            "UPDATE journal_entries AS target SET "
            "previous_hash=(SELECT previous_hash FROM journal_entries WHERE session_id='a' "
            "AND sequence=target.sequence), "
            "record_hash=(SELECT record_hash FROM journal_entries WHERE session_id='a' "
            "AND sequence=target.sequence) WHERE session_id='b'"
        )
    before = _raw_session_rows(journal.path, batches[1].state.session_id)
    attempted = _new_commit_after(
        batches[1].state, version=1, sequence=0 if empty else 2, with_records=False
    )
    for operation in (
        lambda: journal.commit(attempted, 1),
        lambda: journal.load_session(batches[1].state.session_id),
        lambda: journal.load_head(batches[1].state.session_id),
    ):
        with pytest.raises(JournalError) as caught:
            asyncio.run(operation())
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert _raw_session_rows(journal.path, batches[1].state.session_id) == before


def _new_commit_after(
    state: RunState,
    *,
    version: int,
    sequence: int,
    with_records: bool,
) -> JournalCommit:
    projected = replace(
        state,
        version=state.version + 1,
        journal_version=version + 1,
        journal_sequence=sequence + (2 if with_records else 0),
    )
    if not with_records:
        return JournalCommit(projected)
    template = acceptance("template").events[0]
    event = type(template)(
        EventId(f"attempt-event-{version}"),
        template.type,
        projected.run_id,
        projected.session_id,
        template.timestamp,
        {},
    )
    entry = TranscriptEntry(
        projected.session_id,
        projected.run_id,
        projected.submission_id,
        "assistant",
        "attempt",
    )
    return JournalCommit(projected, events=(event,), transcript=(entry,))


@pytest.mark.parametrize("with_records", [False, True], ids=["state-only", "with-records"])
@pytest.mark.parametrize(
    "corruption",
    [
        "delete-event-anchor",
        "delete-event-row",
        "delete-transcript-anchor",
        "delete-transcript-row",
        "wrong-anchor-kind",
        "orphan-event-balanced-row-count",
        "middle-gap-masked-by-cross-table-duplicate",
    ],
)
def test_commit_preflights_existing_session_ledger_before_any_write(
    tmp_path: Path, corruption: str, with_records: bool
) -> None:
    journal = make_journal(tmp_path)
    first = acceptance(f"preflight-{corruption}")
    receipt = asyncio.run(journal.commit(first, 0))
    if corruption == "middle-gap-masked-by-cross-table-duplicate":
        receipt = _append_second_pair(journal, first)
    attempted = _new_commit_after(
        receipt.state,
        version=receipt.version,
        sequence=receipt.sequence,
        with_records=with_records,
    )
    with sqlite3.connect(journal.path) as connection:
        if corruption == "delete-event-anchor":
            connection.execute(
                "DELETE FROM journal_entries WHERE session_id=? AND sequence=1",
                (first.state.session_id.value,),
            )
        elif corruption == "delete-event-row":
            connection.execute(
                "DELETE FROM events WHERE session_id=? AND sequence=1",
                (first.state.session_id.value,),
            )
        elif corruption == "delete-transcript-anchor":
            connection.execute(
                "DELETE FROM journal_entries WHERE session_id=? AND sequence=2",
                (first.state.session_id.value,),
            )
        elif corruption == "delete-transcript-row":
            connection.execute(
                "DELETE FROM transcript WHERE session_id=? AND sequence=2",
                (first.state.session_id.value,),
            )
        elif corruption == "wrong-anchor-kind":
            connection.execute("PRAGMA ignore_check_constraints=ON")
            connection.execute(
                "UPDATE journal_entries SET kind='transcript' WHERE session_id=? AND sequence=1",
                (first.state.session_id.value,),
            )
        elif corruption == "orphan-event-balanced-row-count":
            source = first.events[0]
            orphan_event = type(source)(
                EventId("orphan-event"),
                source.type,
                source.run_id,
                source.session_id,
                source.timestamp,
                source.payload,
            )
            orphan = JournalEvent(2, orphan_event)
            connection.execute(
                "INSERT INTO events(event_id, session_id, sequence, run_id, event_json) "
                "VALUES (?, ?, 2, ?, ?)",
                (
                    orphan.event.event_id.value,
                    first.state.session_id.value,
                    first.state.run_id.value,
                    encode(orphan),
                ),
            )
            connection.execute(
                "DELETE FROM journal_entries WHERE session_id=? AND sequence=1",
                (first.state.session_id.value,),
            )
        else:
            later = receipt.events[-1]
            connection.execute(
                "UPDATE events SET sequence=2, event_json=? WHERE session_id=? AND sequence=3",
                (
                    encode(JournalEvent(2, later.event)),
                    first.state.session_id.value,
                ),
            )
    before = _raw_session_rows(journal.path, first.state.session_id)

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.commit(attempted, receipt.version))

    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert _raw_session_rows(journal.path, first.state.session_id) == before


def test_commit_ledger_preflight_is_session_local(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    damaged = acceptance("damaged", session="damaged")
    healthy = acceptance("healthy", session="healthy")
    asyncio.run(journal.commit(damaged, 0))
    healthy_receipt = asyncio.run(journal.commit(healthy, 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "DELETE FROM journal_entries WHERE session_id=? AND sequence=1",
            (damaged.state.session_id.value,),
        )
    next_healthy = _new_commit_after(
        healthy_receipt.state,
        version=healthy_receipt.version,
        sequence=healthy_receipt.sequence,
        with_records=False,
    )

    receipt = asyncio.run(journal.commit(next_healthy, healthy_receipt.version))

    assert (receipt.version, receipt.sequence) == (2, 2)


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
