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
from src.core.errors import RunErrorCode
from src.core.journal import (
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


@pytest.mark.parametrize("operation", ("reserve", "complete", "abort", "reput"))
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
    elif operation == "reput":
        asyncio.run(journal.complete_artifact_retirement(reservation))
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
        if operation == "reput":
            assert asyncio.run(reopened.artifact_has_active_attachments(ref.artifact_id))
            assert asyncio.run(reopened.load_head(batch.state.session_id)) is not None
        else:
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
