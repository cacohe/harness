from __future__ import annotations

import asyncio
import math
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import TypeVar

from src.adapters.persistence.memory import _canonical_copy, _validate_members
from src.adapters.persistence.sqlite_codec import decode, encode
from src.core.errors import RunErrorCode
from src.core.ids import SessionId, SubmissionId
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalHead,
    JournalSnapshot,
    SubmissionAcceptance,
    TranscriptEntry,
)
from src.core.run_receipt import validate_commit_receipt
from src.core.state import RunState

_SCHEMA_VERSION = 3
_T = TypeVar("_T")

_DDL = """
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  version INTEGER NOT NULL CHECK(version >= 0),
  sequence INTEGER NOT NULL CHECK(sequence >= 0),
  state_json TEXT NOT NULL
) STRICT;
CREATE TABLE journal_entries (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  kind TEXT NOT NULL CHECK(kind IN ('event', 'transcript')),
  PRIMARY KEY(session_id, sequence)
) STRICT;
CREATE TABLE runs (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  status TEXT NOT NULL,
  state_version INTEGER NOT NULL CHECK(state_version >= 0),
  state_json TEXT NOT NULL,
  PRIMARY KEY(session_id, run_id),
  UNIQUE(session_id, submission_id)
) STRICT;
CREATE INDEX runs_active_by_session ON runs(session_id, status);
CREATE TABLE events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  run_id TEXT NOT NULL,
  event_json TEXT NOT NULL,
  UNIQUE(session_id, sequence)
) STRICT;
CREATE INDEX events_by_session_sequence ON events(session_id, sequence);
CREATE TABLE transcript (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  entry_json TEXT NOT NULL,
  PRIMARY KEY(session_id, sequence)
) STRICT;
CREATE INDEX transcript_by_session_sequence ON transcript(session_id, sequence);
CREATE TABLE summary_checkpoints (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  covered_from_sequence INTEGER NOT NULL,
  covered_through_sequence INTEGER NOT NULL,
  policy_version TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  checkpoint_json TEXT NOT NULL,
  PRIMARY KEY(session_id, covered_through_sequence, policy_version, profile_id)
) STRICT;
CREATE TABLE submissions (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  submission_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  digest TEXT NOT NULL,
  intent_json TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  acceptance_state_json TEXT NOT NULL,
  acceptance_version INTEGER NOT NULL,
  PRIMARY KEY(session_id, submission_id)
) STRICT;
CREATE INDEX submissions_by_identity ON submissions(session_id, submission_id, digest);
"""

_EXPECTED_COLUMNS = {
    "sessions": ("session_id", "version", "sequence", "state_json"),
    "journal_entries": ("session_id", "sequence", "kind"),
    "runs": (
        "session_id",
        "run_id",
        "submission_id",
        "profile_id",
        "status",
        "state_version",
        "state_json",
    ),
    "events": ("event_id", "session_id", "sequence", "run_id", "event_json"),
    "transcript": (
        "session_id",
        "sequence",
        "run_id",
        "submission_id",
        "role",
        "content",
        "entry_json",
    ),
    "summary_checkpoints": (
        "session_id",
        "covered_from_sequence",
        "covered_through_sequence",
        "policy_version",
        "profile_id",
        "checkpoint_json",
    ),
    "submissions": (
        "session_id",
        "submission_id",
        "run_id",
        "digest",
        "intent_json",
        "receipt_json",
        "acceptance_state_json",
        "acceptance_version",
    ),
}
_EXPECTED_INDEXES = {
    "runs_active_by_session": ("session_id", "status"),
    "events_by_session_sequence": ("session_id", "sequence"),
    "transcript_by_session_sequence": ("session_id", "sequence"),
    "submissions_by_identity": ("session_id", "submission_id", "digest"),
}


class SQLiteRunJournal:
    """SQLite v3 Session-scoped CAS journal using one connection per operation."""

    def __init__(self, path: Path, timeout_seconds: float = 5.0) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("SQLite journal path must be an absolute Path")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("SQLite journal timeout must be positive and finite")
        if path.exists() and path.is_dir():
            raise ValueError("SQLite journal path cannot be a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self._initialize()

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        if (
            type(batch) is not JournalCommit
            or type(expected_version) is not int
            or expected_version < 0
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        try:
            canonical = _canonical_copy(batch)
        except JournalError:
            raise
        except (MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        return await self._run(
            lambda cancelled: self._commit_sync(canonical, expected_version, cancelled)
        )

    async def load_head(self, session_id: SessionId) -> JournalHead:
        key = self._session_key(session_id)
        return await self._run(lambda _cancelled: self._load_head_sync(key))

    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        key = self._session_key(session_id)
        if type(after_sequence) is not int or after_sequence < 0:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(lambda _cancelled: self._load_session_sync(key, after_sequence))

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionAcceptance | None:
        session_key = self._session_key(session_id)
        if type(submission_id) is not SubmissionId:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        try:
            submission_key = SubmissionId(object.__getattribute__(submission_id, "value")).value
        except (AttributeError, TypeError, ValueError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        return await self._run(
            lambda _cancelled: self._find_submission_sync(session_key, submission_key)
        )

    async def _run(self, operation: Callable[[Event], _T]) -> _T:
        cancelled = Event()
        task = asyncio.create_task(asyncio.to_thread(operation, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            objects = self._application_objects(connection)
            if not objects and version == 0:
                selected = str(
                    connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                ).lower()
                if selected != "wal":
                    raise ValueError("SQLite journal requires WAL mode")
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + _DDL
                    + f"\nPRAGMA user_version = {_SCHEMA_VERSION};\nCOMMIT;"
                )
            elif version != _SCHEMA_VERSION:
                raise ValueError("SQLite journal requires an exact schema v3 database")
            self._validate_schema(connection)
            if str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                raise ValueError("SQLite journal requires WAL mode")
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout_seconds * 1000))}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        selected = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if selected != "wal":
            connection.close()
            raise ValueError("SQLite journal requires WAL mode")
        return connection

    def _commit_sync(
        self, batch: JournalCommit, expected_version: int, cancelled: Event
    ) -> JournalSnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state = batch.state
            session_key = state.session_id.value
            row, _current = _preflight_existing_ledger(connection, session_key)
            version = int(row["version"]) if row is not None else 0
            previous_sequence = int(row["sequence"]) if row is not None else 0
            existing = connection.execute(
                "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
                "acceptance_state_json, acceptance_version FROM submissions "
                "WHERE session_id = ? AND submission_id = ?",
                (session_key, state.submission_id.value),
            ).fetchone()
            intent_json = encode(batch) if batch.submission is not None else None
            if existing is not None and intent_json is not None:
                prior_receipt, _, _ = _decode_submission_row(
                    existing, session_key, state.submission_id.value
                )
                if existing["intent_json"] != intent_json:
                    raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                connection.rollback()
                return prior_receipt
            if existing is not None and intent_json is None:
                prior_receipt, _, _ = _decode_submission_row(
                    existing, session_key, state.submission_id.value
                )
                if prior_receipt.state is None or prior_receipt.state.run_id != state.run_id:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            if version != expected_version:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            _validate_members(batch, previous_sequence, expected_version)
            prior_run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, state_json "
                "FROM runs WHERE session_id = ? AND run_id = ?",
                (session_key, state.run_id.value),
            ).fetchone()
            if prior_run_row is not None:
                prior = decode(prior_run_row["state_json"], RunState)
                _validate_run_row(prior_run_row, prior)
                if (
                    prior.submission_id != state.submission_id
                    or prior.profile_id != state.profile_id
                    or prior.session_id != state.session_id
                    or state.version < prior.version
                ):
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            cursor = previous_sequence
            final_sequence = previous_sequence + len(batch.events) + len(batch.transcript)
            persisted_version = version + 1
            state_json = encode(state)
            connection.execute(
                "INSERT INTO sessions(session_id, version, sequence, state_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET version=excluded.version, sequence=excluded.sequence, state_json=excluded.state_json",
                (session_key, persisted_version, final_sequence, state_json),
            )
            connection.execute(
                "INSERT INTO runs(session_id, run_id, submission_id, profile_id, status, state_version, state_json) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, run_id) DO UPDATE SET status=excluded.status, state_version=excluded.state_version, state_json=excluded.state_json",
                (
                    session_key,
                    state.run_id.value,
                    state.submission_id.value,
                    state.profile_id.value,
                    state.status.value,
                    state.version,
                    state_json,
                ),
            )
            events: list[JournalEvent] = []
            for event in batch.events:
                cursor += 1
                record = JournalEvent(cursor, event)
                connection.execute(
                    "INSERT INTO events(event_id, session_id, sequence, run_id, event_json) VALUES (?, ?, ?, ?, ?)",
                    (event.event_id.value, session_key, cursor, event.run_id.value, encode(record)),
                )
                connection.execute(
                    "INSERT INTO journal_entries(session_id, sequence, kind) VALUES (?, ?, 'event')",
                    (session_key, cursor),
                )
                events.append(record)
            transcript: list[TranscriptEntry] = []
            for entry in batch.transcript:
                cursor += 1
                transcript_record = replace(entry, sequence=cursor)
                connection.execute(
                    "INSERT INTO transcript(session_id, sequence, run_id, submission_id, role, content, entry_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_key,
                        cursor,
                        transcript_record.run_id.value,
                        transcript_record.submission_id.value,
                        transcript_record.role,
                        transcript_record.content,
                        encode(transcript_record),
                    ),
                )
                connection.execute(
                    "INSERT INTO journal_entries(session_id, sequence, kind) VALUES (?, ?, 'transcript')",
                    (session_key, cursor),
                )
                transcript.append(transcript_record)
            receipt = JournalSnapshot(
                state.session_id,
                persisted_version,
                cursor,
                state,
                tuple(events),
                tuple(transcript),
                (batch.summary,) if batch.summary is not None else (),
                batch.submission,
            )
            if batch.summary is not None:
                summary = batch.summary
                summary_json = encode(summary)
                prior_summary = connection.execute(
                    "SELECT session_id, covered_from_sequence, covered_through_sequence, "
                    "policy_version, profile_id, checkpoint_json FROM summary_checkpoints WHERE session_id = ? "
                    "AND covered_through_sequence = ? AND policy_version = ? AND profile_id = ?",
                    (
                        session_key,
                        summary.covered_through_sequence,
                        summary.policy_version,
                        summary.profile_id.value,
                    ),
                ).fetchone()
                if prior_summary is None:
                    connection.execute(
                        "INSERT INTO summary_checkpoints(session_id, covered_from_sequence, covered_through_sequence, policy_version, profile_id, checkpoint_json) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            session_key,
                            summary.covered_from_sequence,
                            summary.covered_through_sequence,
                            summary.policy_version,
                            summary.profile_id.value,
                            summary_json,
                        ),
                    )
                else:
                    persisted_summary = _decode_summary_row(prior_summary, session_key)
                    if (
                        persisted_summary != summary
                        or prior_summary["checkpoint_json"] != summary_json
                    ):
                        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            if batch.submission is not None and intent_json is not None:
                connection.execute(
                    "INSERT INTO submissions(session_id, submission_id, run_id, digest, intent_json, receipt_json, acceptance_state_json, acceptance_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_key,
                        state.submission_id.value,
                        state.run_id.value,
                        batch.submission.digest,
                        intent_json,
                        encode(receipt),
                        state_json,
                        persisted_version,
                    ),
                )
            if cancelled.is_set():
                raise asyncio.CancelledError
            connection.commit()
            return decode(encode(receipt), JournalSnapshot)
        except asyncio.CancelledError:
            connection.rollback()
            raise
        except JournalError:
            connection.rollback()
            raise
        except sqlite3.IntegrityError:
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        except (
            sqlite3.Error,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OverflowError,
            MemoryError,
            RecursionError,
        ):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_head_sync(self, session_key: str) -> JournalHead:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT session_id, version, sequence, state_json FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                return JournalHead(0, 0)
            state = decode(row["state_json"], RunState)
            version, sequence = _validate_session_row(row, session_key, state)
            run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, state_json "
                "FROM runs WHERE session_id = ? AND run_id = ?",
                (session_key, state.run_id.value),
            ).fetchone()
            if run_row is None or decode(run_row["state_json"], RunState) != state:
                raise ValueError
            _validate_run_row(run_row, state)
            return JournalHead(version, sequence, state)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_session_sync(self, session_key: str, after_sequence: int) -> JournalSnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT session_id, version, sequence, state_json FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                connection.commit()
                return JournalSnapshot(SessionId(session_key), 0, 0)
            event_rows = tuple(
                connection.execute(
                    "SELECT event_id, session_id, sequence, run_id, event_json FROM events "
                    "WHERE session_id = ? ORDER BY sequence",
                    (session_key,),
                )
            )
            events = tuple(_decode_event_row(item, session_key) for item in event_rows)
            transcript_rows = tuple(
                connection.execute(
                    "SELECT session_id, sequence, run_id, submission_id, role, content, entry_json "
                    "FROM transcript WHERE session_id = ? ORDER BY sequence",
                    (session_key,),
                )
            )
            transcript = tuple(
                _decode_transcript_row(item, session_key) for item in transcript_rows
            )
            summary_rows = tuple(
                connection.execute(
                    "SELECT session_id, covered_from_sequence, covered_through_sequence, "
                    "policy_version, profile_id, checkpoint_json FROM summary_checkpoints "
                    "WHERE session_id = ? ORDER BY covered_through_sequence, policy_version, profile_id",
                    (session_key,),
                )
            )
            summaries = tuple(_decode_summary_row(item, session_key) for item in summary_rows)
            submission_rows = tuple(
                connection.execute(
                    "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
                    "acceptance_state_json, acceptance_version FROM submissions "
                    "WHERE session_id = ? ORDER BY submission_id",
                    (session_key,),
                )
            )
            acceptance_receipts = tuple(
                _decode_submission_row(item, session_key, str(item["submission_id"]))[0]
                for item in submission_rows
            )
            order_rows = tuple(
                connection.execute(
                    "SELECT sequence, kind FROM journal_entries "
                    "WHERE session_id = ? ORDER BY sequence",
                    (session_key,),
                )
            )
            state = decode(row["state_json"], RunState)
            version, sequence = _validate_session_row(row, session_key, state)
            run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, state_json "
                "FROM runs WHERE session_id = ? AND run_id = ?",
                (session_key, state.run_id.value),
            ).fetchone()
            if run_row is None or decode(run_row["state_json"], RunState) != state:
                raise ValueError
            _validate_run_row(run_row, state)
            shared_sequences = sorted(
                (entry.sequence for entry in events),
                key=int,
            )
            shared_sequences.extend(entry.sequence for entry in transcript)
            shared_sequences.sort()
            if shared_sequences != list(range(1, sequence + 1)):
                raise ValueError
            events_by_sequence = {entry.sequence: entry for entry in events}
            transcript_by_sequence = {entry.sequence: entry for entry in transcript}
            expected_order = {
                **{entry.sequence: "event" for entry in events},
                **{entry.sequence: "transcript" for entry in transcript},
            }
            if (
                len(order_rows) != sequence
                or [row["sequence"] for row in order_rows] != list(range(1, sequence + 1))
                or any(expected_order.get(row["sequence"]) != row["kind"] for row in order_rows)
            ):
                raise ValueError
            for receipt in acceptance_receipts:
                if any(events_by_sequence.get(entry.sequence) != entry for entry in receipt.events):
                    raise ValueError
                if any(
                    transcript_by_sequence.get(entry.sequence) != entry
                    for entry in receipt.transcript
                ):
                    raise ValueError
            events = tuple(entry for entry in events if entry.sequence > after_sequence)
            transcript = tuple(entry for entry in transcript if entry.sequence > after_sequence)
            result = JournalSnapshot(
                SessionId(session_key),
                version,
                sequence,
                state,
                events,
                transcript,
                summaries,
            )
            connection.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _find_submission_sync(
        self, session_key: str, submission_key: str
    ) -> SubmissionAcceptance | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
                "acceptance_state_json, acceptance_version FROM submissions "
                "WHERE session_id = ? AND submission_id = ?",
                (session_key, submission_key),
            ).fetchone()
            if row is None:
                return None
            receipt, state, version = _decode_submission_row(row, session_key, submission_key)
            assert receipt.submission is not None
            return SubmissionAcceptance(receipt.submission, state, version)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    @staticmethod
    def _session_key(session_id: SessionId) -> str:
        if type(session_id) is not SessionId:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        try:
            return SessionId(object.__getattribute__(session_id, "value")).value
        except (AttributeError, TypeError, ValueError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None

    @staticmethod
    def _application_objects(connection: sqlite3.Connection) -> set[tuple[str, str]]:
        return {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
            )
        }

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        if _schema_signature(connection) != _expected_schema_signature():
            raise ValueError("SQLite journal schema v3 definition has drifted")
        for table, expected in _EXPECTED_COLUMNS.items():
            columns = tuple(
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if columns != expected:
                raise ValueError("SQLite journal schema v3 has drifted")
        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            )
        }
        if not set(_EXPECTED_INDEXES) <= indexes:
            raise ValueError("SQLite journal schema v3 is missing required indexes")
        for index, expected_columns in _EXPECTED_INDEXES.items():
            actual_columns = tuple(
                str(row[2]) for row in connection.execute(f"PRAGMA index_info({index})")
            )
            if actual_columns != expected_columns:
                raise ValueError("SQLite journal schema v3 index has drifted")
        for table in (
            "journal_entries",
            "runs",
            "events",
            "transcript",
            "summary_checkpoints",
            "submissions",
        ):
            foreign = tuple(connection.execute(f"PRAGMA foreign_key_list({table})"))
            if not any(
                str(row[2]) == "sessions"
                and str(row[3]) == "session_id"
                and str(row[4]) == "session_id"
                and str(row[6]).upper() == "CASCADE"
                for row in foreign
            ):
                raise ValueError("SQLite journal schema v3 is missing a Session foreign key")


def _schema_signature(connection: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY type, name"
        )
    )


def _expected_schema_signature() -> tuple[tuple[str, str, str, str], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_DDL)
        return _schema_signature(connection)
    finally:
        connection.close()


def _preflight_existing_ledger(
    connection: sqlite3.Connection, session_key: str
) -> tuple[sqlite3.Row | None, RunState | None]:
    """Validate one Session's immutable ledger before a commit reads its CAS counters."""
    row = connection.execute(
        "SELECT session_id, version, sequence, state_json FROM sessions WHERE session_id = ?",
        (session_key,),
    ).fetchone()
    anchor_stats = connection.execute(
        "SELECT COUNT(*), MIN(sequence), MAX(sequence) FROM journal_entries WHERE session_id = ?",
        (session_key,),
    ).fetchone()
    event_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?", (session_key,)
        ).fetchone()[0]
    )
    transcript_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM transcript WHERE session_id = ?", (session_key,)
        ).fetchone()[0]
    )
    anchor_count = int(anchor_stats[0])
    if row is None:
        if anchor_count or event_count or transcript_count:
            raise ValueError("SQLite ledger has records without a Session")
        return None, None

    state = decode(row["state_json"], RunState)
    _, head_sequence = _validate_session_row(row, session_key, state)
    current_run_row = connection.execute(
        "SELECT session_id, run_id, submission_id, profile_id, status, state_version, state_json "
        "FROM runs WHERE session_id = ? AND run_id = ?",
        (session_key, state.run_id.value),
    ).fetchone()
    if current_run_row is None or decode(current_run_row["state_json"], RunState) != state:
        raise ValueError("SQLite Session head disagrees with its Run projection")
    _validate_run_row(current_run_row, state)

    minimum = anchor_stats[1]
    maximum = anchor_stats[2]
    if (
        anchor_count != head_sequence
        or event_count + transcript_count != head_sequence
        or (head_sequence == 0 and (minimum is not None or maximum is not None))
        or (head_sequence > 0 and (minimum != 1 or maximum != head_sequence))
    ):
        raise ValueError("SQLite ledger sequence range disagrees with its Session head")

    bad_anchor = int(
        connection.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM journal_entries AS j "
            "LEFT JOIN events AS e ON e.session_id=j.session_id AND e.sequence=j.sequence "
            "LEFT JOIN transcript AS t ON t.session_id=j.session_id AND t.sequence=j.sequence "
            "WHERE j.session_id=? AND ("
            "j.kind NOT IN ('event', 'transcript') OR "
            "(j.kind='event' AND (e.sequence IS NULL OR t.sequence IS NOT NULL)) OR "
            "(j.kind='transcript' AND (t.sequence IS NULL OR e.sequence IS NOT NULL))"
            ") LIMIT 1)",
            (session_key,),
        ).fetchone()[0]
    )
    orphan_event = int(
        connection.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM events AS e "
            "LEFT JOIN journal_entries AS j "
            "ON j.session_id=e.session_id AND j.sequence=e.sequence AND j.kind='event' "
            "WHERE e.session_id=? AND j.sequence IS NULL LIMIT 1)",
            (session_key,),
        ).fetchone()[0]
    )
    orphan_transcript = int(
        connection.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM transcript AS t "
            "LEFT JOIN journal_entries AS j "
            "ON j.session_id=t.session_id AND j.sequence=t.sequence AND j.kind='transcript' "
            "WHERE t.session_id=? AND j.sequence IS NULL LIMIT 1)",
            (session_key,),
        ).fetchone()[0]
    )
    if bad_anchor or orphan_event or orphan_transcript:
        raise ValueError("SQLite ledger anchors disagree with physical records")

    for event_row in connection.execute(
        "SELECT event_id, session_id, sequence, run_id, event_json FROM events "
        "WHERE session_id = ? ORDER BY sequence",
        (session_key,),
    ):
        _decode_event_row(event_row, session_key)
    for transcript_row in connection.execute(
        "SELECT session_id, sequence, run_id, submission_id, role, content, entry_json "
        "FROM transcript WHERE session_id = ? ORDER BY sequence",
        (session_key,),
    ):
        _decode_transcript_row(transcript_row, session_key)

    for submission_row in connection.execute(
        "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
        "acceptance_state_json, acceptance_version FROM submissions "
        "WHERE session_id = ? ORDER BY submission_id",
        (session_key,),
    ):
        receipt, _, _ = _decode_submission_row(
            submission_row, session_key, str(submission_row["submission_id"])
        )
        for expected_event in receipt.events:
            physical = connection.execute(
                "SELECT event_id, session_id, sequence, run_id, event_json FROM events "
                "WHERE session_id = ? AND sequence = ?",
                (session_key, expected_event.sequence),
            ).fetchone()
            if physical is None or _decode_event_row(physical, session_key) != expected_event:
                raise ValueError("SQLite acceptance Event disagrees with physical history")
        for expected_entry in receipt.transcript:
            physical = connection.execute(
                "SELECT session_id, sequence, run_id, submission_id, role, content, entry_json "
                "FROM transcript WHERE session_id = ? AND sequence = ?",
                (session_key, expected_entry.sequence),
            ).fetchone()
            if physical is None or _decode_transcript_row(physical, session_key) != expected_entry:
                raise ValueError("SQLite acceptance transcript disagrees with physical history")
    return row, state


def _validate_session_row(row: sqlite3.Row, session_key: str, state: RunState) -> tuple[int, int]:
    version = row["version"]
    sequence = row["sequence"]
    if (
        row["session_id"] != session_key
        or state.session_id.value != session_key
        or type(version) is not int
        or version < 1
        or type(sequence) is not int
        or sequence < 0
        or state.journal_version != version
        or state.journal_sequence != sequence
    ):
        raise ValueError("SQLite Session row disagrees with its state")
    return version, sequence


def _validate_run_row(row: sqlite3.Row, state: RunState) -> None:
    if (
        row["session_id"] != state.session_id.value
        or row["run_id"] != state.run_id.value
        or row["submission_id"] != state.submission_id.value
        or row["profile_id"] != state.profile_id.value
        or row["status"] != state.status.value
        or type(row["state_version"]) is not int
        or row["state_version"] != state.version
    ):
        raise ValueError("SQLite Run row disagrees with its state")


def _decode_event_row(row: sqlite3.Row, session_key: str) -> JournalEvent:
    record = decode(row["event_json"], JournalEvent)
    if (
        row["session_id"] != session_key
        or type(row["sequence"]) is not int
        or row["sequence"] != record.sequence
        or row["event_id"] != record.event.event_id.value
        or row["run_id"] != record.event.run_id.value
        or record.event.session_id.value != session_key
    ):
        raise ValueError("SQLite Event row disagrees with its payload")
    return record


def _decode_transcript_row(row: sqlite3.Row, session_key: str) -> TranscriptEntry:
    record = decode(row["entry_json"], TranscriptEntry)
    if (
        row["session_id"] != session_key
        or type(row["sequence"]) is not int
        or row["sequence"] != record.sequence
        or row["run_id"] != record.run_id.value
        or row["submission_id"] != record.submission_id.value
        or row["role"] != record.role
        or row["content"] != record.content
        or record.session_id.value != session_key
    ):
        raise ValueError("SQLite Transcript row disagrees with its payload")
    return record


def _decode_summary_row(row: sqlite3.Row, session_key: str) -> ConversationSummaryCheckpoint:
    record = decode(row["checkpoint_json"], ConversationSummaryCheckpoint)
    if (
        row["session_id"] != session_key
        or row["covered_from_sequence"] != record.covered_from_sequence
        or row["covered_through_sequence"] != record.covered_through_sequence
        or row["policy_version"] != record.policy_version
        or row["profile_id"] != record.profile_id.value
        or record.session_id.value != session_key
    ):
        raise ValueError("SQLite Summary row disagrees with its payload")
    return record


def _decode_submission_row(
    row: sqlite3.Row, session_key: str, submission_key: str
) -> tuple[JournalSnapshot, RunState, int]:
    receipt = decode(row["receipt_json"], JournalSnapshot)
    state = decode(row["acceptance_state_json"], RunState)
    intent = decode(row["intent_json"], JournalCommit)
    version = row["acceptance_version"]
    submission = receipt.submission
    span = len(intent.events) + len(intent.transcript)
    previous_sequence = receipt.sequence - span
    if previous_sequence < 0:
        raise ValueError("SQLite Submission receipt has an invalid sequence range")
    validate_commit_receipt(
        receipt,
        intent,
        expected_version=version - 1,
        previous_sequence=previous_sequence,
    )
    if (
        submission is None
        or type(version) is not int
        or version < 1
        or receipt.version != version
        or receipt.session_id.value != session_key
        or receipt.state != state
        or state.session_id.value != session_key
        or state.submission_id.value != submission_key
        or intent.state != state
        or intent.submission != submission
        or state.journal_version != version
        or state.journal_sequence != receipt.sequence
        or row["session_id"] != session_key
        or row["submission_id"] != submission_key
        or row["run_id"] != state.run_id.value
        or row["digest"] != submission.digest
        or submission.session_id.value != session_key
        or submission.run_id != state.run_id
        or submission.submission_id.value != submission_key
    ):
        raise ValueError("SQLite Submission row disagrees with its receipt")
    return receipt, state, version
