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
from src.core.ids import EventId, ProfileId, RunId, SessionId
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalSnapshot,
    TranscriptEntry,
    TranscriptEntryType,
)
from src.core.state import RunState, RunStatus
from tests.unit.core.test_journal_contract import acceptance, artifact_ref
from tests.unit.core.test_run_state import (
    advance,
    ask_approval,
    begin_step,
    decide,
    initial,
    submit,
)


def make_journal(tmp_path: Path) -> SQLiteRunJournal:
    return SQLiteRunJournal((tmp_path / "journal.sqlite3").resolve())


@pytest.mark.parametrize("attack", ("fact", "active-update", "active-delete", "active-insert"))
def test_artifact_fact_and_active_index_tamper_is_rejected(tmp_path: Path, attack: str) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance(f"artifact-tamper-{attack}")
    fact = ArtifactAttachmentFact(
        batch.state.session_id,
        batch.state.run_id,
        batch.state.submission_id,
        "owner",
        ArtifactAttachmentRole.TOOL_RESULT,
        ArtifactAttachmentAction.ATTACHED,
        artifact_ref("d"),
    )
    asyncio.run(journal.commit(replace(batch, artifact_facts=(fact,)), 0))
    with sqlite3.connect(journal.path) as connection:
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        if attack == "fact":
            connection.execute("DELETE FROM artifact_attachment_facts")
        elif attack == "active-update":
            connection.execute("UPDATE artifact_attachments SET owner_id='moved'")
        elif attack == "active-delete":
            connection.execute("DELETE FROM artifact_attachments")
        else:
            connection.execute(
                "INSERT INTO artifact_attachments(session_id, run_id, submission_id, owner_id, role, artifact_id, fact_json) "
                "SELECT session_id, run_id, submission_id, 'forged', role, artifact_id, fact_json "
                "FROM artifact_attachments"
            )

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_artifact_attachments(batch.state.session_id, batch.state.run_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    with pytest.raises(JournalError) as retirement:
        asyncio.run(journal.artifact_has_active_attachments(fact.artifact.artifact_id))
    assert retirement.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def _coverage_history(journal: SQLiteRunJournal):
    batch = acceptance("sticky-coverage")
    receipt = None
    for version in range(2):
        if version:
            assert receipt is not None
            batch = _new_commit_after(receipt.state, version=1, sequence=201, with_records=True)
        entries = tuple(
            TranscriptEntry(
                batch.state.session_id,
                batch.state.run_id,
                batch.state.submission_id,
                "assistant",
                f"history-{version}-{index}",
            )
            for index in range(200)
        )
        receipt = asyncio.run(
            journal.commit(
                replace(
                    batch,
                    state=replace(batch.state, journal_sequence=(version + 1) * 201),
                    submission=None,
                    summary=None,
                    transcript=entries,
                ),
                version,
            )
        )
    assert receipt is not None
    return receipt


@pytest.mark.parametrize("attack", ("delete", "update", "replace", "replace-rowid"))
def test_integrity_violation_cannot_be_cleared_with_ordinary_dml(tmp_path, attack):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE journal_entries SET record_hash=? WHERE sequence=200", ("0" * 64,)
        )
        try:
            if attack == "delete":
                connection.execute("DELETE FROM integrity_violations")
            elif attack == "update":
                connection.execute("UPDATE integrity_violations SET session_id='other'")
            elif attack == "replace-rowid":
                connection.execute(
                    "INSERT OR REPLACE INTO integrity_violations(rowid, session_id, reason) "
                    "SELECT rowid, 'other', reason FROM integrity_violations"
                )
            else:
                connection.execute(
                    "INSERT OR REPLACE INTO integrity_violations VALUES ('session', 'replacement')"
                )
                connection.execute("DELETE FROM integrity_violations")
        except sqlite3.DatabaseError:
            pass
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.summarize_before(receipt.session_id, 403))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize("table", ("transcript", "journal_entries"))
@pytest.mark.parametrize("connection_kind", ("external", "reader"))
def test_direct_insert_cannot_bypass_full_middle_block(tmp_path, table, connection_kind):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    blocked = False
    with (
        sqlite3.connect(journal.path) if connection_kind == "external" else journal._connect()
    ) as connection:
        try:
            if table == "transcript":
                # Sequence 202 is an event inside the second complete transcript block.
                entry = replace(receipt.transcript[0], sequence=202)
                connection.execute(
                    "INSERT INTO transcript VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.session_id.value,
                        202,
                        entry.run_id.value,
                        entry.submission_id.value,
                        entry.role,
                        entry.content,
                        entry.entry_type.value,
                        entry.format_version,
                        encode(entry),
                    ),
                )
            else:
                connection.execute(
                    "INSERT INTO journal_entries VALUES ('session', 403, 'event', ?, ?)",
                    ("0" * 64, "0" * 64),
                )
        except sqlite3.DatabaseError:
            # A connection without the adapter's append capability must fail closed.
            blocked = True
    if blocked:
        coverage = asyncio.run(journal.summarize_before(receipt.session_id, 403))
        assert coverage.segments[0].item_count == 400
        assert asyncio.run(journal.read_page(receipt.session_id, None, 2)).interactions
    else:
        with pytest.raises(JournalError):
            asyncio.run(journal.summarize_before(receipt.session_id, 403))
        with pytest.raises(JournalError):
            asyncio.run(journal.read_page(receipt.session_id, None, 2))


@pytest.mark.parametrize("table", ("transcript", "journal_entries"))
@pytest.mark.parametrize("session", ("session", "destination"))
def test_cross_session_update_taints_both_sessions(tmp_path, table, session):
    journal = make_journal(tmp_path)
    _coverage_history(journal)
    asyncio.run(journal.commit(acceptance("dest", session="destination"), 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            f"UPDATE {table} SET session_id='destination' WHERE session_id='session' AND sequence=200"
        )
    with pytest.raises(JournalError):
        asyncio.run(journal.summarize_before(SessionId(session), 403))


@pytest.mark.parametrize("table", ("transcript", "journal_entries"))
def test_reader_connection_cannot_replace_another_sessions_hidden_rowid(tmp_path, table):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    asyncio.run(journal.commit(acceptance("dest", session="destination"), 0))
    blocked = False
    with journal._connect() as connection:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        projection = ["'destination'" if column == "session_id" else column for column in columns]
        try:
            connection.execute(
                f"INSERT OR REPLACE INTO {table}(rowid, {', '.join(columns)}) "
                f"SELECT rowid, {', '.join(projection)} FROM {table} "
                "WHERE session_id='session' AND sequence=200"
            )
        except sqlite3.DatabaseError:
            blocked = True
    if blocked:
        assert (
            asyncio.run(journal.summarize_before(receipt.session_id, 403)).segments[0].item_count
            == 400
        )
    else:
        with pytest.raises(JournalError):
            asyncio.run(journal.summarize_before(receipt.session_id, 403))


def test_read_page_rejects_known_middle_corruption(tmp_path):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(
            "UPDATE journal_entries SET record_hash=? WHERE sequence=200", ("0" * 64,)
        )
    with pytest.raises(JournalError):
        asyncio.run(journal.read_page(receipt.session_id, None, 2))


def test_read_page_does_not_publish_unauthenticated_remaining_omission(tmp_path):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    page = asyncio.run(journal.read_page(receipt.session_id, None, 2))
    assert page.next_before_sequence == 401
    assert page.remaining_omission is None


@pytest.mark.parametrize("table", ("transcript", "journal_entries"))
def test_delete_taints_history_and_rollback_restores_clean_commit_channel(tmp_path, table):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    with sqlite3.connect(journal.path) as connection:
        connection.execute(f"DELETE FROM {table} WHERE sequence=200")
        connection.rollback()
    assert (
        asyncio.run(journal.summarize_before(receipt.session_id, 403)).segments[0].item_count == 400
    )
    next_batch = _new_commit_after(receipt.state, version=2, sequence=402, with_records=True)
    asyncio.run(journal.commit(next_batch, 2))
    reopened = make_journal(tmp_path)
    assert (
        asyncio.run(reopened.summarize_before(receipt.session_id, 405)).segments[0].item_count
        == 401
    )
    with sqlite3.connect(journal.path) as connection:
        connection.execute(f"DELETE FROM {table} WHERE sequence=200")
    with pytest.raises(JournalError):
        asyncio.run(reopened.read_page(receipt.session_id, None, 2))
    with pytest.raises(JournalError):
        asyncio.run(reopened.summarize_before(receipt.session_id, 405))


def test_sqlite_aggregate_vm_work_is_bounded_as_history_grows(
    tmp_path, monkeypatch, record_property
):
    journal = make_journal(tmp_path)
    batch = acceptance("vm-work")
    steps = 0
    real_connect = journal._connect

    def measured_connect():
        connection = real_connect()

        def tick():
            nonlocal steps
            steps += 100
            return 0

        connection.set_progress_handler(tick, 100)
        return connection

    async def scenario():
        nonlocal steps
        counts = []
        total = 0
        for version, count in enumerate((400, *([960] * 10), *([1024] * 10))):
            total += count
            transcript = tuple(
                replace(batch.transcript[0], content=f"row-{total}-{index}")
                for index in range(count)
            )
            await journal.commit(
                replace(
                    batch,
                    state=replace(
                        batch.state, journal_version=version + 1, journal_sequence=total + 1
                    ),
                    events=batch.events if version == 0 else (),
                    submission=None,
                    summary=None,
                    transcript=transcript,
                ),
                version,
            )
            if total in (400, 10_000, 20_240):
                monkeypatch.setattr(journal, "_connect", measured_connect)
                steps = 0
                coverage = await journal.summarize_before(batch.state.session_id, total + 2)
                assert coverage.segments[0].item_count == total
                counts.append(steps)
                monkeypatch.setattr(journal, "_connect", real_connect)
        return counts

    counts = asyncio.run(scenario())
    record_property("sqlite_vm_steps_400_10000_20240", counts)
    # All sizes have a 16-row final block. Additional interior blocks must be
    # summarized via the fixed-depth tree, never visited as physical history.
    assert max(counts) <= min(counts) + 10_000, counts


def test_coverage_of_event_only_gap_inside_transcript_block_is_explicitly_empty(tmp_path):
    journal = make_journal(tmp_path)
    receipt = _coverage_history(journal)
    coverage = asyncio.run(journal.summarize_before(receipt.session_id, 203, 1, 201))
    assert [
        (segment.first_sequence, segment.last_sequence, segment.item_count)
        for segment in coverage.segments
    ] == [(202, 202, 0)]


@pytest.mark.parametrize("corruption", ("record-hash", "payload"))
def test_conversation_aggregate_rejects_authenticated_middle_corruption(
    tmp_path: Path, corruption: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("aggregate-corruption")
    transcript = tuple(
        TranscriptEntry(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "assistant",
            f"history-{index}",
        )
        for index in range(20)
    )
    commit = replace(
        batch,
        state=replace(batch.state, journal_sequence=21),
        submission=None,
        transcript=transcript,
        summary=None,
    )
    asyncio.run(journal.commit(commit, 0))
    with sqlite3.connect(journal.path) as connection:
        if corruption == "record-hash":
            connection.execute(
                "UPDATE journal_entries SET record_hash=? WHERE session_id=? AND sequence=10",
                ("0" * 64, batch.state.session_id.value),
            )
        else:
            changed = replace(transcript[8], sequence=10, content="tampered")
            connection.execute(
                "UPDATE transcript SET content=?, entry_json=? WHERE session_id=? AND sequence=10",
                (changed.content, encode(changed), batch.state.session_id.value),
            )

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.summarize_before(batch.state.session_id, 22))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize(
    "corruption",
    ("middle-record", "middle-payload", "block", "node", "session-root"),
)
def test_conversation_aggregate_rejects_full_block_and_witness_corruption(
    tmp_path: Path, corruption: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("coverage-witness")
    transcript = tuple(
        TranscriptEntry(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "assistant",
            f"history-{index}",
        )
        for index in range(400)
    )
    asyncio.run(
        journal.commit(
            replace(
                batch,
                state=replace(batch.state, journal_sequence=401),
                submission=None,
                transcript=transcript,
                summary=None,
            ),
            0,
        )
    )
    with sqlite3.connect(journal.path) as connection:
        if corruption == "middle-record":
            connection.execute(
                "UPDATE journal_entries SET record_hash=? WHERE session_id=? AND sequence=200",
                ("0" * 64, batch.state.session_id.value),
            )
        elif corruption == "middle-payload":
            changed = replace(transcript[198], sequence=200, content="tampered")
            connection.execute(
                "UPDATE transcript SET content=?, entry_json=? WHERE session_id=? AND sequence=200",
                (changed.content, encode(changed), batch.state.session_id.value),
            )
        elif corruption == "block":
            connection.execute(
                "UPDATE transcript_coverage_blocks SET content_hash=? "
                "WHERE session_id=? AND block_index=1",
                ("0" * 64, batch.state.session_id.value),
            )
        elif corruption == "node":
            connection.execute(
                "UPDATE transcript_coverage_nodes SET digest=? "
                "WHERE session_id=? AND level=1 AND node_index=0",
                ("0" * 64, batch.state.session_id.value),
            )
        else:
            connection.execute(
                "UPDATE sessions SET authenticated_root=? WHERE session_id=?",
                ("0" * 64, batch.state.session_id.value),
            )

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.summarize_before(batch.state.session_id, 402))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_sqlite_v6_round_trips_and_reopens(tmp_path: Path) -> None:
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


def test_context_summary_lookup_filters_bounds_and_survives_restart(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("summary-restart")
    first_receipt = asyncio.run(journal.commit(first, 0))
    second = _new_commit_after(
        first_receipt.state,
        version=first_receipt.version,
        sequence=first_receipt.sequence,
        with_records=True,
    )
    second_summary = ConversationSummaryCheckpoint(
        second.state.session_id,
        second.state.journal_sequence,
        "v1",
        second.state.profile_id,
        "second",
    )
    second_receipt = asyncio.run(
        journal.commit(replace(second, summary=second_summary), first_receipt.version)
    )
    third = _new_commit_after(
        second_receipt.state,
        version=second_receipt.version,
        sequence=second_receipt.sequence,
        with_records=True,
    )
    third_summary = ConversationSummaryCheckpoint(
        third.state.session_id,
        third.state.journal_sequence,
        "v2",
        third.state.profile_id,
        "third",
    )
    receipt = asyncio.run(
        journal.commit(replace(third, summary=third_summary), second_receipt.version)
    )
    reopened = make_journal(tmp_path)

    assert asyncio.run(
        reopened.load_context_summaries(
            receipt.session_id,
            receipt.state.profile_id,
            "v1",
            receipt.sequence,
            128,
            receipt.version,
            receipt.sequence,
        )
    ) == (second_summary, first.summary)
    assert asyncio.run(
        reopened.load_context_summaries(
            receipt.session_id,
            receipt.state.profile_id,
            "v1",
            first_receipt.sequence,
            128,
            receipt.version,
            receipt.sequence,
        )
    ) == (first.summary,)
    assert (
        asyncio.run(
            reopened.load_context_summaries(
                receipt.session_id,
                ProfileId("other"),
                "v1",
                receipt.sequence,
                128,
                receipt.version,
                receipt.sequence,
            )
        )
        == ()
    )


@pytest.mark.parametrize("table", ("summary_checkpoints", "summary_nodes"))
def test_context_summary_lookup_rejects_known_summary_tampering(tmp_path: Path, table: str) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("summary-tamper")
    receipt = asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        connection.execute(f"DELETE FROM {table} WHERE session_id=?", (receipt.session_id.value,))

    with pytest.raises(JournalError) as caught:
        asyncio.run(
            journal.load_context_summaries(
                receipt.session_id,
                receipt.state.profile_id,
                "v1",
                receipt.sequence,
                128,
                receipt.version,
                receipt.sequence,
            )
        )
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def _large_transcript_history(journal: SQLiteRunJournal, total: int = 20_000) -> JournalSnapshot:
    template = acceptance("summary-scale")
    state = template.state
    receipt = None
    sequence = 0
    for version, chunk_start in enumerate(range(0, total, 4_000)):
        chunk_size = min(4_000, total - chunk_start)
        events = template.events if version == 0 else ()
        transcript_count = chunk_size - len(events)
        transcript = tuple(
            replace(template.transcript[0], content=f"history-{chunk_start + index}")
            for index in range(transcript_count)
        )
        sequence += chunk_size
        state = replace(
            state,
            version=state.version + (1 if version else 0),
            journal_version=version + 1,
            journal_sequence=sequence,
        )
        receipt = asyncio.run(
            journal.commit(
                JournalCommit(state, events=events, transcript=transcript),
                version,
            )
        )
    assert receipt is not None
    return receipt


def _append_summary_fixture(
    journal: SQLiteRunJournal,
    receipt: JournalSnapshot,
    through_count: int,
) -> None:
    session_key = receipt.session_id.value
    with journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
        row = connection.execute(
            "SELECT ledger_hash, coverage_hash, transcript_count, summary_count, "
            "session_state_hash, run_projection_hash, run_count "
            "FROM sessions WHERE session_id=?",
            (session_key,),
        ).fetchone()
        assert row is not None
        start = int(row["summary_count"])
        summary_hash = sqlite_module.summary_empty_hash(session_key, 32, 0)
        if start:
            summary_hash = connection.execute(
                "SELECT summary_hash FROM sessions WHERE session_id=?", (session_key,)
            ).fetchone()[0]
        for summary_index in range(start, through_count):
            checkpoint = ConversationSummaryCheckpoint(
                receipt.session_id,
                summary_index + 1,
                "v-scale",
                receipt.state.profile_id,
                f"summary-{summary_index}",
            )
            checkpoint_json = encode(checkpoint)
            checkpoint_hash = sqlite_module.summary_record_hash(
                session_key, summary_index, checkpoint_json
            )
            connection.execute(
                "INSERT INTO summary_checkpoints(session_id, summary_index, "
                "covered_from_sequence, covered_through_sequence, policy_version, profile_id, "
                "checkpoint_json, checkpoint_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_key,
                    summary_index,
                    checkpoint.covered_from_sequence,
                    checkpoint.covered_through_sequence,
                    checkpoint.policy_version,
                    checkpoint.profile_id.value,
                    checkpoint_json,
                    checkpoint_hash,
                ),
            )
            summary_hash = sqlite_module._append_summary_leaf(
                connection, session_key, summary_index, checkpoint_hash
            )
        authenticated_root = sqlite_module.authenticated_root_hash(
            session_key,
            row["ledger_hash"],
            row["coverage_hash"],
            row["transcript_count"],
            summary_hash,
            through_count,
            row["session_state_hash"],
            row["run_projection_hash"],
            row["run_count"],
        )
        connection.execute(
            "UPDATE sessions SET summary_hash=?, summary_count=?, authenticated_root=? "
            "WHERE session_id=?",
            (summary_hash, through_count, authenticated_root, session_key),
        )


def test_context_summary_lookup_vm_work_and_decoding_are_near_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property
) -> None:
    journal = make_journal(tmp_path)
    receipt = _large_transcript_history(journal)
    real_connect = journal._connect
    real_decode = sqlite_module.decode
    measuring = False
    steps = 0
    decoded_summaries = 0

    def measured_connect():
        connection = real_connect()

        def tick():
            nonlocal steps
            if measuring:
                steps += 100
            return 0

        connection.set_progress_handler(tick, 100)
        return connection

    def measured_decode(payload, expected_type):
        nonlocal decoded_summaries
        if measuring and expected_type is ConversationSummaryCheckpoint:
            decoded_summaries += 1
        return real_decode(payload, expected_type)

    monkeypatch.setattr(journal, "_connect", measured_connect)
    monkeypatch.setattr(sqlite_module, "decode", measured_decode)
    samples = []
    for count in (400, 10_000, 20_000):
        _append_summary_fixture(journal, receipt, count)
        steps = 0
        decoded_summaries = 0
        measuring = True
        selected = asyncio.run(
            journal.load_context_summaries(
                receipt.session_id,
                receipt.state.profile_id,
                "v-scale",
                receipt.sequence,
                128,
                receipt.version,
                receipt.sequence,
            )
        )
        measuring = False
        assert len(selected) == 128
        assert decoded_summaries == 128
        assert selected[0].covered_through_sequence == count
        samples.append(steps)
    record_property("context_summary_lookup_vm_steps", samples)
    assert max(samples) <= min(samples) * 1.35 + 1_000


def test_sqlite_reopen_preserves_structured_multi_tool_interaction(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("structured-tools")
    assistant = TranscriptEntry(
        batch.state.session_id,
        batch.state.run_id,
        batch.state.submission_id,
        "assistant",
        '{"text":"checking","tool_calls":['
        '{"id":"a","name":"alpha","arguments":"{}"},'
        '{"id":"b","name":"beta","arguments":"{}"}]}',
        entry_type=TranscriptEntryType.ASSISTANT_TOOL_BATCH,
        format_version=1,
    )
    results = (
        TranscriptEntry(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "tool",
            '{"tool_call_id":"b","result":2}',
            entry_type=TranscriptEntryType.TOOL_RESULT,
            format_version=1,
        ),
        TranscriptEntry(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "tool",
            '{"tool_call_id":"a","result":1}',
            entry_type=TranscriptEntryType.TOOL_RESULT,
            format_version=1,
        ),
    )
    committed = replace(
        batch,
        state=replace(batch.state, journal_sequence=4),
        transcript=(assistant, *results),
    )

    asyncio.run(journal.commit(committed, 0))
    page = asyncio.run(make_journal(tmp_path).read_page(batch.state.session_id, None, 1))

    assert len(page.interactions) == 1
    assert [message.role.value for message in page.interactions[0].messages] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert [message.tool_call_id for message in page.interactions[0].messages[1:]] == ["b", "a"]


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


def test_schema_is_exact_v6_and_has_required_indexes(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    with sqlite3.connect(journal.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
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
            "run_projection_nodes",
            "events",
            "submissions",
            "transcript",
            "transcript_coverage_blocks",
            "transcript_coverage_nodes",
            "integrity_violations",
            "summary_checkpoints",
            "summary_nodes",
            "artifact_attachment_facts",
            "artifact_attachments",
            "artifact_retirements",
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
            "transcript_coverage_blocks_by_sequence",
            "summary_context_lookup",
            "artifact_attachments_by_artifact",
        } <= indexes


@pytest.mark.parametrize("version", [1, 2, 4, 5, 7])
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

    with pytest.raises(ValueError, match="schema v6"):
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
            "  authenticated_root TEXT NOT NULL\n",
            "  authenticated_root TEXT\n",
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
        connection.execute("PRAGMA user_version=5")
    with pytest.raises(ValueError, match="schema v6"):
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


def test_synchronized_session_and_run_state_payload_tamper_is_rejected(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("synchronized-state")
    asyncio.run(journal.commit(batch, 0))
    forged = replace(batch.state, elapsed_seconds=0.5)
    try:
        with sqlite3.connect(journal.path) as connection:
            connection.execute("UPDATE sessions SET state_json=?", (encode(forged),))
            connection.execute("UPDATE runs SET state_json=?", (encode(forged),))
    except sqlite3.OperationalError:
        return

    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_head(batch.state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


@pytest.mark.parametrize("attack", ("payload", "state_hash", "node", "root", "count"))
def test_historical_run_projection_point_proof_rejects_tamper(tmp_path: Path, attack: str) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("history-first")
    asyncio.run(journal.commit(first, 0))
    second = acceptance("history-second")
    second = replace(
        second,
        state=replace(second.state, journal_version=2, journal_sequence=4),
        summary=None,
    )
    asyncio.run(journal.commit(second, 1))
    with sqlite3.connect(journal.path) as connection:
        if attack == "payload":
            forged = replace(first.state, elapsed_seconds=0.5)
            connection.execute(
                "UPDATE runs SET state_json=? WHERE run_id=?",
                (encode(forged), first.state.run_id.value),
            )
        elif attack == "state_hash":
            connection.execute(
                "UPDATE runs SET state_hash=? WHERE run_id=?",
                ("0" * 64, first.state.run_id.value),
            )
        elif attack == "node":
            connection.execute(
                "UPDATE run_projection_nodes SET digest=? WHERE level=0 AND node_index=0",
                ("0" * 64,),
            )
        elif attack == "root":
            connection.execute("UPDATE sessions SET run_projection_hash=?", ("0" * 64,))
        else:
            connection.execute("UPDATE sessions SET run_count=99")

    with pytest.raises(JournalError) as caught:
        asyncio.run(
            journal.find_run_state(
                first.state.session_id, first.state.run_id, first.state.submission_id
            )
        )
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_synchronized_approval_binding_tamper_is_rejected(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    state = advance(initial(), submit(), *begin_step(), ask_approval(), decide(True))
    state = replace(state, journal_version=1, journal_sequence=0)
    asyncio.run(journal.commit(JournalCommit(state), 0))
    step = state.current_step
    assert step is not None and step.tool_calls and step.tool_calls[0].approval is not None
    tool = replace(
        step.tool_calls[0], approval=replace(step.tool_calls[0].approval, approved=False)
    )
    forged = replace(state, current_step=replace(step, tool_calls=(tool,)))
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE sessions SET state_json=?", (encode(forged),))
        connection.execute("UPDATE runs SET state_json=?", (encode(forged),))
    with pytest.raises(JournalError) as caught:
        asyncio.run(journal.load_head(state.session_id))
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_run_projection_cannot_be_moved_across_sessions(tmp_path: Path) -> None:
    journal = make_journal(tmp_path)
    first = acceptance("cross-one", session="one")
    second = acceptance("cross-two", session="two")
    asyncio.run(journal.commit(first, 0))
    asyncio.run(journal.commit(second, 0))
    with sqlite3.connect(journal.path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE runs SET session_id=? WHERE session_id=? AND run_id=?",
            (second.state.session_id.value, first.state.session_id.value, first.state.run_id.value),
        )


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
                # Offline damage fixture explicitly bypasses append authorization
                # to keep testing hash validation independently of the SQL guard.
                connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
                connection.execute(
                    "INSERT INTO transcript(session_id, sequence, run_id, submission_id, role, "
                    "content, entry_type, format_version, entry_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        extra.session_id.value,
                        5,
                        extra.run_id.value,
                        extra.submission_id.value,
                        extra.role,
                        extra.content,
                        extra.entry_type.value,
                        extra.format_version,
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


@pytest.mark.parametrize("corruption", ["row", "payload", "identity"])
def test_find_run_input_uses_identity_index_and_rejects_tampered_row(
    tmp_path: Path, corruption: str
) -> None:
    journal = make_journal(tmp_path)
    batch = acceptance("run-input")
    asyncio.run(journal.commit(batch, 0))
    with sqlite3.connect(journal.path) as connection:
        plan = tuple(
            connection.execute(
                "EXPLAIN QUERY PLAN SELECT session_id, sequence, run_id, submission_id, role, "
                "content, entry_json FROM transcript WHERE session_id=? AND run_id=? "
                "AND submission_id=? AND role='user' ORDER BY sequence LIMIT 2",
                (
                    batch.state.session_id.value,
                    batch.state.run_id.value,
                    batch.state.submission_id.value,
                ),
            )
        )
        assert any("transcript_run_input" in str(row[3]) for row in plan)
        identity = (
            batch.state.session_id.value,
            batch.state.run_id.value,
            batch.state.submission_id.value,
        )
        if corruption == "row":
            connection.execute(
                "UPDATE transcript SET content='tampered' WHERE session_id=? AND run_id=? "
                "AND submission_id=? AND role='user'",
                identity,
            )
        elif corruption == "payload":
            changed = replace(batch.transcript[0], content="tampered", sequence=2)
            connection.execute(
                "UPDATE transcript SET content=?, entry_json=? WHERE session_id=? AND run_id=? "
                "AND submission_id=? AND role='user'",
                (changed.content, encode(changed), *identity),
            )
        else:
            changed = replace(batch.transcript[0], run_id=RunId("other"), sequence=2)
            connection.execute(
                "UPDATE transcript SET entry_json=? WHERE session_id=? AND run_id=? "
                "AND submission_id=? AND role='user'",
                (encode(changed), *identity),
            )

    with pytest.raises(JournalError) as caught:
        asyncio.run(
            journal.find_run_input(
                batch.state.session_id,
                batch.state.run_id,
                batch.state.submission_id,
            )
        )
    assert caught.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


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
