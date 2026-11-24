import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
import src.adapters.persistence.sqlite as sqlite_module
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.adapters.persistence.sqlite_integrity import retirement_empty_hash
from src.core.execution.errors import RunErrorCode
from src.core.persistence.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ArtifactRetirementState,
    JournalError,
)
from tests.unit.core.test_journal_contract import acceptance, artifact_ref


def database_dump(path: Path) -> tuple[str, ...]:
    with closing(sqlite3.connect(path)) as connection:
        return tuple(connection.iterdump())


@pytest.mark.parametrize("missing", ((1,), (0, 2), (1, 2)))
def test_new_retirement_rejects_missing_current_rows(tmp_path: Path, missing: tuple[int, ...]):
    journal = SQLiteRunJournal((tmp_path / "occupancy.sqlite3").resolve())
    for seed in "abcd":
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref(seed).artifact_id))
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        connection.executemany(
            "DELETE FROM artifact_retirements WHERE artifact_index=?", [(i,) for i in missing]
        )
    before = database_dump(journal.path)
    for candidate in (journal, SQLiteRunJournal(journal.path)):
        with pytest.raises(JournalError) as caught:
            asyncio.run(candidate.reserve_artifact_retirement(artifact_ref("e").artifact_id))
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert database_dump(journal.path) == before


@pytest.mark.parametrize("operation", ("update", "replace", "upsert"))
def test_retirement_index_rewrites_cannot_hide_allocation_damage(tmp_path: Path, operation: str):
    journal = SQLiteRunJournal((tmp_path / "rewritten.sqlite3").resolve())
    for seed in "abc":
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref(seed).artifact_id))
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        if operation == "update":
            connection.execute(
                "UPDATE artifact_retirements SET artifact_index=4 WHERE artifact_index=1"
            )
            connection.execute(
                "UPDATE artifact_retirements SET artifact_index=1 WHERE artifact_index=4"
            )
        else:
            # REPLACE's implicit DELETE does not run DELETE triggers with recursive_triggers OFF.
            assert connection.execute("PRAGMA recursive_triggers").fetchone() == (0,)
            command = (
                "INSERT OR REPLACE INTO artifact_retirements "
                if operation == "replace"
                else "INSERT INTO artifact_retirements "
            )
            command += (
                "SELECT artifact_id, 2, generation, state, fact_count, fact_root, state_hash "
                "FROM artifact_retirements WHERE artifact_index=1"
            )
            if operation == "upsert":
                command += " ON CONFLICT(artifact_id) DO UPDATE SET artifact_index=4"
            connection.execute(command)
            if operation == "upsert":
                connection.execute(
                    "UPDATE artifact_retirements SET artifact_index=1 WHERE artifact_index=4"
                )
    before = database_dump(journal.path)
    for candidate in (journal, SQLiteRunJournal(journal.path)):
        with pytest.raises(JournalError) as caught:
            asyncio.run(candidate.reserve_artifact_retirement(artifact_ref("d").artifact_id))
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert database_dump(journal.path) == before


def test_retirement_artifact_id_replace_cannot_hide_middle_gap(tmp_path: Path):
    journal = SQLiteRunJournal((tmp_path / "artifact-id-replace.sqlite3").resolve())
    for seed in "abc":
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref(seed).artifact_id))
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        assert connection.execute("PRAGMA recursive_triggers").fetchone() == (0,)
        connection.execute(
            "UPDATE OR REPLACE artifact_retirements "
            "SET artifact_id=(SELECT artifact_id FROM artifact_retirements WHERE artifact_index=1) "
            "WHERE artifact_index=2"
        )
        assert connection.execute(
            "SELECT artifact_index FROM artifact_retirements ORDER BY artifact_index"
        ).fetchall() == [(0,), (2,)]
    before = database_dump(journal.path)
    for candidate in (journal, SQLiteRunJournal(journal.path)):
        with pytest.raises(JournalError) as caught:
            asyncio.run(candidate.reserve_artifact_retirement(artifact_ref("d").artifact_id))
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert database_dump(journal.path) == before


def test_retirement_allocation_damage_cannot_be_cleared_with_write_capability(tmp_path: Path):
    journal = SQLiteRunJournal((tmp_path / "permanent.sqlite3").resolve())
    asyncio.run(journal.reserve_artifact_retirement(artifact_ref("a").artifact_id))
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        connection.execute("DELETE FROM artifact_retirements")
    before = database_dump(journal.path)
    with closing(sqlite3.connect(journal.path)) as connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        for statement in (
            "DELETE FROM artifact_retirement_allocation_damage",
            "UPDATE artifact_retirement_allocation_damage SET singleton=1",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="damage is permanent"):
                connection.execute(statement)
            connection.rollback()
        # REPLACE cannot erase the only legal singleton, even without recursive triggers.
        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifact_retirement_allocation_damage VALUES (1)"
            )
    assert database_dump(journal.path) == before
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref("b").artifact_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize(
    "target", ("artifact_retirements_delete_damage", "artifact_retirement_damage_delete_guard")
)
def test_retirement_allocation_witness_schema_is_authenticated(tmp_path: Path, target: str):
    journal = SQLiteRunJournal((tmp_path / "schema.sqlite3").resolve())
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute(f"DROP TRIGGER {target}")
    with pytest.raises(ValueError, match="definition has drifted"):
        SQLiteRunJournal(journal.path)


def test_rolled_back_retirement_damage_preserves_future_allocation(tmp_path: Path):
    journal = SQLiteRunJournal((tmp_path / "rollback.sqlite3").resolve())
    for seed in "abc":
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref(seed).artifact_id))
    before = database_dump(journal.path)
    with closing(sqlite3.connect(journal.path)) as connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        connection.execute("DELETE FROM artifact_retirements WHERE artifact_index=1")
        assert connection.execute(
            "SELECT singleton FROM artifact_retirement_allocation_damage"
        ).fetchall() == [(1,)]
        connection.rollback()
    assert database_dump(journal.path) == before
    reopened = SQLiteRunJournal(journal.path)
    assert (
        asyncio.run(reopened.reserve_artifact_retirement(artifact_ref("d").artifact_id)) is not None
    )


@pytest.mark.parametrize("operation", ("complete", "abort", "cross-session-attach"))
def test_current_row_damage_rejects_other_operations_atomically(tmp_path: Path, operation: str):
    journal = SQLiteRunJournal((tmp_path / "global-damage.sqlite3").resolve())
    reservations = [
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref(seed).artifact_id))
        for seed in "abc"
    ]
    reservation = reservations[0]
    assert reservation is not None
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        connection.execute("DELETE FROM artifact_retirements WHERE artifact_index=1")
    before = database_dump(journal.path)
    with pytest.raises(JournalError) as caught:
        if operation == "complete":
            asyncio.run(journal.complete_artifact_retirement(reservation))
        elif operation == "abort":
            asyncio.run(journal.abort_artifact_retirement(reservation))
        else:
            batch = acceptance("unrelated-session")
            fact = ArtifactAttachmentFact(
                batch.state.session_id,
                batch.state.run_id,
                batch.state.submission_id,
                "owner",
                ArtifactAttachmentRole.FINAL_OUTPUT,
                ArtifactAttachmentAction.ATTACHED,
                artifact_ref("d"),
            )
            asyncio.run(journal.commit(replace(batch, artifact_facts=(fact,)), 0))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert database_dump(journal.path) == before


def test_retirement_occupancy_queries_are_indexed_and_bounded(tmp_path: Path):
    journal = SQLiteRunJournal((tmp_path / "query-plan.sqlite3").resolve())
    steps = []
    for total in (4, 64):

        async def populate():
            for i in range(0 if total == 4 else 4, total):
                await journal.reserve_artifact_retirement(f"sha256:{i:064x}")

        asyncio.run(populate())
        with closing(sqlite3.connect(journal.path)) as connection:
            connection.row_factory = sqlite3.Row
            statements: list[str] = []
            ticks = []
            connection.set_trace_callback(statements.append)
            connection.set_progress_handler(lambda: ticks.append(1) or 0, 1)
            root, count = sqlite_module._verify_retirement_meta(connection)
            sqlite_module._verify_retirement_allocation(connection, root, count)
            connection.set_progress_handler(None, 0)
            connection.set_trace_callback(None)
            steps.append(len(ticks))
            assert not any("COUNT(" in sql.upper() for sql in statements)
            for sql in statements:
                plan = " ".join(
                    str(row[3]) for row in connection.execute("EXPLAIN QUERY PLAN " + sql)
                )
                if "ORDER BY artifact_index" in sql:
                    assert "USING COVERING INDEX" in plan
                    assert "LIMIT 1" in sql
                else:
                    assert "SEARCH" in plan
                    assert "SCAN" not in plan
    # Fixed-depth proof plus singleton/tail seeks: VM work is independent of row count.
    assert max(steps) < 4_000
    assert abs(steps[1] - steps[0]) < 100


@pytest.mark.parametrize("level", range(32))
@pytest.mark.parametrize("attack", ("orphan", "wrong-empty", "empty-with-count", "overflow"))
def test_new_retirement_rejects_unauthenticated_empty_path(
    tmp_path: Path, level: int, attack: str
) -> None:
    journal = SQLiteRunJournal((tmp_path / "empty-path.sqlite3").resolve())
    original = asyncio.run(journal.reserve_artifact_retirement(artifact_ref("a").artifact_id))
    assert original is not None
    # Leaf 1 is the next allocation; at every higher level node 1 is its empty sibling.
    digest, count = "0" * 64, 1
    if attack == "wrong-empty":
        count = 0
    elif attack == "empty-with-count":
        digest = retirement_empty_hash(level, 1)
    elif attack == "overflow":
        count = (1 << level) + 1
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        connection.execute(
            "INSERT INTO artifact_retirement_nodes VALUES (?, 1, ?, ?)",
            (level, digest, count),
        )
    before = database_dump(journal.path)
    for _ in range(2):
        reopened = SQLiteRunJournal(journal.path)
        with pytest.raises(JournalError) as caught:
            asyncio.run(reopened.reserve_artifact_retirement(artifact_ref("b").artifact_id))
        assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert database_dump(journal.path) == before


@pytest.mark.parametrize("level", range(33))
def test_new_retirement_rejects_missing_old_path_nodes(tmp_path: Path, level: int) -> None:
    journal = SQLiteRunJournal((tmp_path / "missing-path.sqlite3").resolve())
    asyncio.run(journal.reserve_artifact_retirement(artifact_ref("a").artifact_id))
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        connection.execute("DELETE FROM artifact_retirement_nodes WHERE level=?", (level,))
    before = database_dump(journal.path)
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref("b").artifact_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert database_dump(journal.path) == before


@pytest.mark.parametrize(
    "attack", ("distant-orphan", "out-of-tree", "index-gap", "root", "count", "meta")
)
def test_new_retirement_rejects_frontier_corruption(tmp_path: Path, attack: str) -> None:
    journal = SQLiteRunJournal((tmp_path / "frontier.sqlite3").resolve())
    asyncio.run(journal.reserve_artifact_retirement(artifact_ref("a").artifact_id))
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        if attack in ("distant-orphan", "out-of-tree"):
            index = 7 if attack == "distant-orphan" else 1 << 32
            connection.execute(
                "INSERT INTO artifact_retirement_nodes VALUES (0, ?, ?, 1)",
                (index, "0" * 64),
            )
        elif attack == "index-gap":
            connection.execute("UPDATE artifact_retirements SET artifact_index=3")
        elif attack == "root":
            connection.execute(
                "UPDATE artifact_retirement_nodes SET digest=? WHERE level=32", ("0" * 64,)
            )
        elif attack == "count":
            connection.execute("UPDATE artifact_retirement_meta SET artifact_count=2")
        else:
            connection.execute("DELETE FROM artifact_retirement_meta")
    before = database_dump(journal.path)
    with pytest.raises(JournalError) as caught:
        asyncio.run(
            SQLiteRunJournal(journal.path).reserve_artifact_retirement(
                artifact_ref("b").artifact_id
            )
        )
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert database_dump(journal.path) == before


@pytest.mark.parametrize("operation", ("reserve", "complete", "abort"))
@pytest.mark.parametrize("room", (0, 1))
def test_retirement_transition_limit_is_checked_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, room: int
) -> None:
    journal = SQLiteRunJournal((tmp_path / "limit.sqlite3").resolve())
    ref = artifact_ref("a")
    reservation = asyncio.run(journal.reserve_artifact_retirement(ref.artifact_id))
    assert reservation is not None
    count = 1
    if operation == "reserve":
        asyncio.run(journal.abort_artifact_retirement(reservation))
        count = 2
    monkeypatch.setattr(sqlite_module, "_MAX_ARTIFACT_FACTS_PER_LOOKUP", count + room)
    batch = acceptance("bounded-reput")
    fact = ArtifactAttachmentFact(
        batch.state.session_id,
        batch.state.run_id,
        batch.state.submission_id,
        "owner",
        ArtifactAttachmentRole.FINAL_OUTPUT,
        ArtifactAttachmentAction.ATTACHED,
        ref,
    )

    def append(journal: SQLiteRunJournal):
        if operation == "reserve":
            return asyncio.run(journal.reserve_artifact_retirement(ref.artifact_id))
        if operation == "complete":
            return asyncio.run(journal.complete_artifact_retirement(reservation))
        if operation == "abort":
            return asyncio.run(journal.abort_artifact_retirement(reservation))
        return asyncio.run(journal.commit(replace(batch, artifact_facts=(fact,)), 0))

    if room:
        append(journal)
        with closing(sqlite3.connect(journal.path)) as connection:
            assert connection.execute("SELECT fact_count FROM artifact_retirements").fetchone() == (
                count + 1,
            )
        # Reopening at the exact limit must remain readable and support idempotent operations.
        reopened = SQLiteRunJournal(journal.path)
        assert append(reopened) is not None
    else:
        before = database_dump(journal.path)
        for _ in range(2):
            with pytest.raises(JournalError) as caught:
                append(SQLiteRunJournal(journal.path))
            assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
            assert database_dump(journal.path) == before
        # A rejection leaves valid facts; raising the limit permits that same transition.
        monkeypatch.setattr(sqlite_module, "_MAX_ARTIFACT_FACTS_PER_LOOKUP", count + 1)
        append(SQLiteRunJournal(journal.path))


def test_two_new_retirements_allocate_distinct_authenticated_indices(tmp_path: Path) -> None:
    path = (tmp_path / "concurrent.sqlite3").resolve()
    journals = (SQLiteRunJournal(path), SQLiteRunJournal(path))
    barrier = Barrier(2)

    def reserve(index: int):
        barrier.wait()
        return asyncio.run(
            journals[index].reserve_artifact_retirement(artifact_ref("ab"[index]).artifact_id)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reservations = tuple(executor.map(reserve, (0, 1)))
    reopened = SQLiteRunJournal(path)
    for reservation in reservations:
        assert reservation is not None
        assert asyncio.run(reopened.reserve_artifact_retirement(reservation.artifact_id)) == (
            reservation
        )
        assert asyncio.run(reopened.complete_artifact_retirement(reservation)).state is (
            ArtifactRetirementState.RETIRED
        )
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute(
            "SELECT artifact_index FROM artifact_retirements ORDER BY artifact_index"
        ).fetchall() == [(0,), (1,)]
        assert connection.execute(
            "SELECT artifact_count FROM artifact_retirement_meta"
        ).fetchone() == (2,)


def test_retirement_tree_capacity_does_not_commit_an_unreadable_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite_module, "_RETIREMENT_DEPTH", 2)
    journal = SQLiteRunJournal((tmp_path / "capacity.sqlite3").resolve())
    reservations = [
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref(seed).artifact_id))
        for seed in "abcd"
    ]
    before = database_dump(journal.path)
    with pytest.raises(JournalError):
        asyncio.run(journal.reserve_artifact_retirement(artifact_ref("e").artifact_id))
    assert database_dump(journal.path) == before
    reopened = SQLiteRunJournal(journal.path)
    for reservation in reservations:
        assert reservation is not None
        assert asyncio.run(reopened.complete_artifact_retirement(reservation)).state is (
            ArtifactRetirementState.RETIRED
        )
