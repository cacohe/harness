from __future__ import annotations

import asyncio
import math
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import TypeVar

from src.adapters.context.journal_history import transcript_interaction
from src.adapters.persistence.memory import (
    _canonical_copy,
    _validate_aggregate_request,
    _validate_members,
)
from src.adapters.persistence.sqlite_codec import decode, encode
from src.adapters.persistence.sqlite_integrity import genesis_hash, record_hash, validate_hash
from src.core.context import ConversationPage, OmittedContextRef
from src.core.errors import RunErrorCode
from src.core.ids import RunId, SessionId, SubmissionId
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
_IdT = TypeVar("_IdT", SessionId, RunId, SubmissionId)

_DDL = """
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  version INTEGER NOT NULL CHECK(version >= 0),
  sequence INTEGER NOT NULL CHECK(sequence >= 0),
  state_json TEXT NOT NULL,
  ledger_hash TEXT NOT NULL
) STRICT;
CREATE TABLE journal_entries (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  kind TEXT NOT NULL CHECK(kind IN ('event', 'transcript')),
  previous_hash TEXT NOT NULL,
  record_hash TEXT NOT NULL,
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
CREATE INDEX transcript_run_input ON transcript(session_id, run_id, submission_id, role, sequence);
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
    "sessions": ("session_id", "version", "sequence", "state_json", "ledger_hash"),
    "journal_entries": ("session_id", "sequence", "kind", "previous_hash", "record_hash"),
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
    "transcript_run_input": (
        "session_id",
        "run_id",
        "submission_id",
        "role",
        "sequence",
    ),
    "submissions_by_identity": ("session_id", "submission_id", "digest"),
}


def _typed_id_key(value: object, expected: type[_IdT]) -> str:
    try:
        if type(value) is not expected:
            raise ValueError
        raw = object.__getattribute__(value, "value")
        if type(raw) is not str:
            raise ValueError
        canonical = expected(raw)
        return object.__getattribute__(canonical, "value")
    except (AttributeError, TypeError, ValueError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


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

    async def read_page(
        self, session_id: SessionId, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        key = self._session_key(session_id)
        if (
            before_sequence is not None
            and (type(before_sequence) is not int or before_sequence <= 0)
            or type(limit) is not int
            or not 1 <= limit <= 1_024
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(lambda _cancelled: self._read_page_sync(key, before_sequence, limit))

    async def summarize_before(
        self,
        session_id: SessionId,
        before_sequence: int,
        excluded_from_sequence: int | None = None,
        excluded_through_sequence: int | None = None,
    ) -> tuple[OmittedContextRef, ...]:
        _validate_aggregate_request(
            session_id,
            before_sequence,
            excluded_from_sequence,
            excluded_through_sequence,
        )
        key = self._session_key(session_id)
        return await self._run(
            lambda _cancelled: self._summarize_before_sync(
                key,
                before_sequence,
                excluded_from_sequence,
                excluded_through_sequence,
            )
        )

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionAcceptance | None:
        session_key = self._session_key(session_id)
        submission_key = _typed_id_key(submission_id, SubmissionId)
        return await self._run(
            lambda _cancelled: self._find_submission_sync(session_key, submission_key)
        )

    async def find_run_input(
        self,
        session_id: SessionId,
        run_id: RunId,
        submission_id: SubmissionId,
    ) -> TranscriptEntry | None:
        session_key = self._session_key(session_id)
        run_key = _typed_id_key(run_id, RunId)
        submission_key = _typed_id_key(submission_id, SubmissionId)
        return await self._run(
            lambda _cancelled: self._find_run_input_sync(session_key, run_key, submission_key)
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
            ledger_hash = row["ledger_hash"] if row is not None else genesis_hash(session_key)
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
                "INSERT INTO sessions(session_id, version, sequence, state_json, ledger_hash) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET version=excluded.version, sequence=excluded.sequence, state_json=excluded.state_json",
                (session_key, persisted_version, final_sequence, state_json, ledger_hash),
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
                record_json = encode(record)
                connection.execute(
                    "INSERT INTO events(event_id, session_id, sequence, run_id, event_json) VALUES (?, ?, ?, ?, ?)",
                    (event.event_id.value, session_key, cursor, event.run_id.value, record_json),
                )
                ledger_hash = _append_anchor(
                    connection, session_key, cursor, "event", record_json, ledger_hash
                )
                events.append(record)
            transcript: list[TranscriptEntry] = []
            for entry in batch.transcript:
                cursor += 1
                transcript_record = replace(entry, sequence=cursor)
                record_json = encode(transcript_record)
                connection.execute(
                    "INSERT INTO transcript(session_id, sequence, run_id, submission_id, role, content, entry_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_key,
                        cursor,
                        transcript_record.run_id.value,
                        transcript_record.submission_id.value,
                        transcript_record.role,
                        transcript_record.content,
                        record_json,
                    ),
                )
                ledger_hash = _append_anchor(
                    connection, session_key, cursor, "transcript", record_json, ledger_hash
                )
                transcript.append(transcript_record)
            connection.execute(
                "UPDATE sessions SET ledger_hash=? WHERE session_id=?", (ledger_hash, session_key)
            )
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
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT session_id, version, sequence, state_json, ledger_hash "
                "FROM sessions WHERE session_id = ?",
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
            _verify_ledger_tail(connection, session_key, sequence, row["ledger_hash"])
            return JournalHead(version, sequence, state)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_session_sync(self, session_key: str, after_sequence: int) -> JournalSnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row, state = _preflight_existing_ledger(connection, session_key)
            if row is None:
                connection.commit()
                return JournalSnapshot(SessionId(session_key), 0, 0)
            event_rows = tuple(
                connection.execute(
                    "SELECT event_id, session_id, sequence, run_id, event_json FROM events "
                    "WHERE session_id = ? AND sequence > ? ORDER BY sequence",
                    (session_key, after_sequence),
                )
            )
            events = tuple(_decode_event_row(item, session_key) for item in event_rows)
            transcript_rows = tuple(
                connection.execute(
                    "SELECT session_id, sequence, run_id, submission_id, role, content, entry_json "
                    "FROM transcript WHERE session_id = ? AND sequence > ? ORDER BY sequence",
                    (session_key, after_sequence),
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
            assert state is not None
            version, sequence = _validate_session_row(row, session_key, state)
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

    def _read_page_sync(
        self, session_key: str, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            if before_sequence is None:
                rows = tuple(
                    connection.execute(
                        "SELECT session_id, sequence, run_id, submission_id, role, content, "
                        "entry_json FROM transcript WHERE session_id=? "
                        "ORDER BY sequence DESC LIMIT ?",
                        (session_key, limit + 1),
                    )
                )
            else:
                rows = tuple(
                    connection.execute(
                        "SELECT session_id, sequence, run_id, submission_id, role, content, "
                        "entry_json FROM transcript WHERE session_id=? AND sequence<? "
                        "ORDER BY sequence DESC LIMIT ?",
                        (session_key, before_sequence, limit + 1),
                    )
                )
            selected = rows[:limit]
            entries = tuple(_decode_transcript_row(row, session_key) for row in selected)
            for entry in entries:
                _verify_point_anchor(connection, session_key, entry.sequence)
            cursor = entries[-1].sequence if len(rows) > limit else None
            remaining = (
                _aggregate_transcript_range(connection, session_key, 1, cursor)
                if cursor is not None
                else None
            )
            result = ConversationPage(
                tuple(transcript_interaction(entry) for entry in entries),
                cursor,
                remaining,
            )
            connection.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _summarize_before_sync(
        self,
        session_key: str,
        before_sequence: int,
        excluded_from_sequence: int | None,
        excluded_through_sequence: int | None,
    ) -> tuple[OmittedContextRef, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            result: tuple[OmittedContextRef, ...]
            if excluded_from_sequence is None:
                omission = _aggregate_transcript_range(connection, session_key, 1, before_sequence)
                result = (omission,) if omission is not None else ()
            else:
                assert excluded_through_sequence is not None
                result = tuple(
                    item
                    for item in (
                        _aggregate_transcript_range(
                            connection,
                            session_key,
                            excluded_through_sequence + 1,
                            before_sequence,
                        ),
                        _aggregate_transcript_range(
                            connection,
                            session_key,
                            1,
                            min(excluded_from_sequence, before_sequence),
                        ),
                    )
                    if item is not None
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
            connection.execute("BEGIN")
            _preflight_existing_ledger(connection, session_key)
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

    def _find_run_input_sync(
        self, session_key: str, run_key: str, submission_key: str
    ) -> TranscriptEntry | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = tuple(
                connection.execute(
                    "SELECT session_id, sequence, run_id, submission_id, role, content, "
                    "entry_json FROM transcript WHERE session_id=? AND run_id=? "
                    "AND submission_id=? AND role='user' ORDER BY sequence LIMIT 2",
                    (session_key, run_key, submission_key),
                )
            )
            if len(rows) > 1:
                raise ValueError("SQLite Run has duplicate user inputs")
            if not rows:
                return None
            entry = _decode_transcript_row(rows[0], session_key)
            if (
                entry.run_id.value != run_key
                or entry.submission_id.value != submission_key
                or entry.role != "user"
            ):
                raise ValueError("SQLite Run input identity disagrees")
            run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, "
                "state_json FROM runs WHERE session_id=? AND run_id=?",
                (session_key, run_key),
            ).fetchone()
            if run_row is None:
                raise ValueError("SQLite Run input has no Run projection")
            state = decode(run_row["state_json"], RunState)
            _validate_run_row(run_row, state)
            if state.submission_id.value != submission_key:
                raise ValueError("SQLite Run input disagrees with Run projection")
            _verify_point_anchor(connection, session_key, entry.sequence)
            return entry
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    @staticmethod
    def _session_key(session_id: SessionId) -> str:
        return _typed_id_key(session_id, SessionId)

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


_CHAIN_ROWS = (
    "SELECT j.session_id, j.sequence, j.kind, j.previous_hash, j.record_hash, "
    "e.event_id, e.event_json, t.entry_json, t.submission_id, t.role, t.content, "
    "CASE j.kind WHEN 'event' THEN e.run_id ELSE t.run_id END AS run_id "
    "FROM journal_entries AS j "
    "LEFT JOIN events AS e ON e.session_id=j.session_id AND e.sequence=j.sequence "
    "LEFT JOIN transcript AS t ON t.session_id=j.session_id AND t.sequence=j.sequence"
)


def _append_anchor(
    connection: sqlite3.Connection,
    session_key: str,
    sequence: int,
    kind: str,
    canonical_json: str,
    previous_hash: str,
) -> str:
    digest = record_hash(session_key, sequence, kind, canonical_json, previous_hash)
    connection.execute(
        "INSERT INTO journal_entries(session_id, sequence, kind, previous_hash, record_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_key, sequence, kind, previous_hash, digest),
    )
    return digest


def _verify_anchor(anchor: sqlite3.Row, session_key: str, previous_hash: str) -> str:
    if validate_hash(anchor["previous_hash"]) != previous_hash:
        raise ValueError("SQLite ledger predecessor hash disagrees")
    if anchor["kind"] == "event" and anchor["entry_json"] is None:
        _decode_event_row(anchor, session_key)
        canonical_json = anchor["event_json"]
    elif anchor["kind"] == "transcript" and anchor["event_json"] is None:
        _decode_transcript_row(anchor, session_key)
        canonical_json = anchor["entry_json"]
    else:
        raise ValueError("SQLite ledger anchor kind disagrees with its record")
    # The decoders reject non-canonical JSON before the stored text is hashed.
    digest = record_hash(
        session_key, anchor["sequence"], anchor["kind"], canonical_json, previous_hash
    )
    if validate_hash(anchor["record_hash"]) != digest:
        raise ValueError("SQLite ledger record hash disagrees with its payload")
    return digest


def _verify_ledger_tail(
    connection: sqlite3.Connection, session_key: str, sequence: int, ledger_hash: object
) -> None:
    """Check the head boundary in O(1) indexed reads, without authenticating older history."""
    root = validate_hash(ledger_hash)
    previous_hash = genesis_hash(session_key)
    if sequence == 0:
        if root != previous_hash:
            raise ValueError("SQLite empty ledger root disagrees with its Session")
        return
    if sequence > 1:
        previous = connection.execute(
            "SELECT record_hash FROM journal_entries WHERE session_id=? AND sequence=?",
            (session_key, sequence - 1),
        ).fetchone()
        if previous is None:
            raise ValueError("SQLite ledger tail predecessor is missing")
        previous_hash = validate_hash(previous["record_hash"])
    anchor = connection.execute(
        _CHAIN_ROWS + " WHERE j.session_id=? AND j.sequence=?", (session_key, sequence)
    ).fetchone()
    if anchor is None or _verify_anchor(anchor, session_key, previous_hash) != root:
        raise ValueError("SQLite ledger tail disagrees with its Session root")


def _verify_point_anchor(connection: sqlite3.Connection, session_key: str, sequence: int) -> None:
    """Authenticate one indexed record and its immediate ledger boundary in O(1)."""
    session = connection.execute(
        "SELECT sequence, ledger_hash FROM sessions WHERE session_id=?", (session_key,)
    ).fetchone()
    if session is None or type(session["sequence"]) is not int or sequence > session["sequence"]:
        raise ValueError("SQLite point anchor is outside its Session")
    previous_hash = genesis_hash(session_key)
    if sequence > 1:
        previous = connection.execute(
            "SELECT record_hash FROM journal_entries WHERE session_id=? AND sequence=?",
            (session_key, sequence - 1),
        ).fetchone()
        if previous is None:
            raise ValueError("SQLite point anchor predecessor is missing")
        previous_hash = validate_hash(previous["record_hash"])
    anchor = connection.execute(
        _CHAIN_ROWS + " WHERE j.session_id=? AND j.sequence=?", (session_key, sequence)
    ).fetchone()
    if anchor is None:
        raise ValueError("SQLite point anchor is missing")
    digest = _verify_anchor(anchor, session_key, previous_hash)
    if sequence == session["sequence"] and digest != validate_hash(session["ledger_hash"]):
        raise ValueError("SQLite point anchor disagrees with Session root")


def _aggregate_transcript_range(
    connection: sqlite3.Connection,
    session_key: str,
    first_inclusive: int,
    before_sequence: int,
) -> OmittedContextRef | None:
    if first_inclusive >= before_sequence:
        return None
    row = connection.execute(
        "SELECT COUNT(*) AS item_count, MIN(sequence) AS first_sequence, "
        "MAX(sequence) AS last_sequence FROM transcript "
        "WHERE session_id=? AND sequence>=? AND sequence<?",
        (session_key, first_inclusive, before_sequence),
    ).fetchone()
    if row is None or type(row["item_count"]) is not int:
        raise ValueError("SQLite transcript aggregate is invalid")
    count = row["item_count"]
    if count == 0:
        if row["first_sequence"] is not None or row["last_sequence"] is not None:
            raise ValueError("SQLite empty transcript aggregate has boundaries")
        return None
    first = row["first_sequence"]
    last = row["last_sequence"]
    if (
        type(first) is not int
        or type(last) is not int
        or not first_inclusive <= first <= last < before_sequence
        or count > last - first + 1
    ):
        raise ValueError("SQLite transcript aggregate disagrees with its range")
    _verify_point_anchor(connection, session_key, first)
    if last != first:
        _verify_point_anchor(connection, session_key, last)
    return OmittedContextRef("transcript", first, last, count)


def _preflight_existing_ledger(
    connection: sqlite3.Connection, session_key: str
) -> tuple[sqlite3.Row | None, RunState | None]:
    """Validate one Session's immutable ledger before a commit reads its CAS counters."""
    row = connection.execute(
        "SELECT session_id, version, sequence, state_json, ledger_hash "
        "FROM sessions WHERE session_id = ?",
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

    previous_hash = genesis_hash(session_key)
    for anchor in connection.execute(
        _CHAIN_ROWS + " WHERE j.session_id=? ORDER BY j.sequence", (session_key,)
    ):
        previous_hash = _verify_anchor(anchor, session_key, previous_hash)
    if previous_hash != validate_hash(row["ledger_hash"]):
        raise ValueError("SQLite ledger root disagrees with its history")

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
