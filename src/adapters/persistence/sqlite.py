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
CREATE TABLE runs (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  status TEXT NOT NULL,
  state_json TEXT NOT NULL,
  PRIMARY KEY(session_id, run_id),
  UNIQUE(session_id, submission_id)
) STRICT;
CREATE INDEX runs_active_by_session ON runs(session_id, status);
CREATE TABLE events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
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
    "runs": ("session_id", "run_id", "submission_id", "profile_id", "status", "state_json"),
    "events": ("event_id", "session_id", "sequence", "event_json"),
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
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = self._application_tables(connection)
            if not tables and version == 0:
                connection.executescript(_DDL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION or tables != set(_EXPECTED_COLUMNS):
                raise ValueError("SQLite journal requires an exact schema v3 database")
            self._validate_schema(connection)
            connection.commit()
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
        selected = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
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
            row = connection.execute(
                "SELECT version, sequence, state_json FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            version = int(row["version"]) if row is not None else 0
            previous_sequence = int(row["sequence"]) if row is not None else 0
            existing = connection.execute(
                "SELECT intent_json, receipt_json FROM submissions WHERE session_id = ? AND submission_id = ?",
                (session_key, state.submission_id.value),
            ).fetchone()
            intent_json = encode(batch) if batch.submission is not None else None
            if existing is not None and intent_json is not None:
                if existing["intent_json"] != intent_json:
                    raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                receipt = decode(existing["receipt_json"], JournalSnapshot)
                connection.rollback()
                return receipt
            if existing is not None and intent_json is None:
                prior_receipt = decode(existing["receipt_json"], JournalSnapshot)
                if prior_receipt.state is None or prior_receipt.state.run_id != state.run_id:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            if version != expected_version:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            _validate_members(batch, previous_sequence)
            prior_run_row = connection.execute(
                "SELECT state_json FROM runs WHERE session_id = ? AND run_id = ?",
                (session_key, state.run_id.value),
            ).fetchone()
            if prior_run_row is not None:
                prior = decode(prior_run_row["state_json"], RunState)
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
                "INSERT INTO runs(session_id, run_id, submission_id, profile_id, status, state_json) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, run_id) DO UPDATE SET status=excluded.status, state_json=excluded.state_json",
                (
                    session_key,
                    state.run_id.value,
                    state.submission_id.value,
                    state.profile_id.value,
                    state.status.value,
                    state_json,
                ),
            )
            events: list[JournalEvent] = []
            for event in batch.events:
                cursor += 1
                record = JournalEvent(cursor, event)
                connection.execute(
                    "INSERT INTO events(event_id, session_id, sequence, event_json) VALUES (?, ?, ?, ?)",
                    (event.event_id.value, session_key, cursor, encode(record)),
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
                connection.execute(
                    "INSERT INTO summary_checkpoints(session_id, covered_through_sequence, policy_version, profile_id, checkpoint_json) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id, covered_through_sequence, policy_version, profile_id) DO UPDATE SET checkpoint_json=excluded.checkpoint_json",
                    (
                        session_key,
                        summary.covered_through_sequence,
                        summary.policy_version,
                        summary.profile_id.value,
                        encode(summary),
                    ),
                )
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
        except (sqlite3.Error, ValueError, TypeError, AttributeError, KeyError, OverflowError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_head_sync(self, session_key: str) -> JournalHead:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT version, sequence, state_json FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                return JournalHead(0, 0)
            return JournalHead(
                int(row["version"]), int(row["sequence"]), decode(row["state_json"], RunState)
            )
        except (sqlite3.Error, ValueError, TypeError, KeyError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_session_sync(self, session_key: str, after_sequence: int) -> JournalSnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT version, sequence, state_json FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                connection.commit()
                return JournalSnapshot(SessionId(session_key), 0, 0)
            events = tuple(
                decode(item["event_json"], JournalEvent)
                for item in connection.execute(
                    "SELECT event_json FROM events WHERE session_id = ? AND sequence > ? ORDER BY sequence",
                    (session_key, after_sequence),
                )
            )
            transcript = tuple(
                decode(item["entry_json"], TranscriptEntry)
                for item in connection.execute(
                    "SELECT entry_json FROM transcript WHERE session_id = ? AND sequence > ? ORDER BY sequence",
                    (session_key, after_sequence),
                )
            )
            summaries = tuple(
                decode(item["checkpoint_json"], ConversationSummaryCheckpoint)
                for item in connection.execute(
                    "SELECT checkpoint_json FROM summary_checkpoints WHERE session_id = ? ORDER BY covered_through_sequence, policy_version, profile_id",
                    (session_key,),
                )
            )
            result = JournalSnapshot(
                SessionId(session_key),
                int(row["version"]),
                int(row["sequence"]),
                decode(row["state_json"], RunState),
                events,
                transcript,
                summaries,
            )
            connection.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError, KeyError):
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
                "SELECT receipt_json, acceptance_state_json, acceptance_version FROM submissions WHERE session_id = ? AND submission_id = ?",
                (session_key, submission_key),
            ).fetchone()
            if row is None:
                return None
            receipt = decode(row["receipt_json"], JournalSnapshot)
            if receipt.submission is None:
                raise ValueError
            return SubmissionAcceptance(
                receipt.submission,
                decode(row["acceptance_state_json"], RunState),
                int(row["acceptance_version"]),
            )
        except (sqlite3.Error, ValueError, TypeError, KeyError):
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
    def _application_tables(connection: sqlite3.Connection) -> set[str]:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
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
        for table in ("runs", "events", "transcript", "summary_checkpoints", "submissions"):
            foreign = tuple(connection.execute(f"PRAGMA foreign_key_list({table})"))
            if not any(
                str(row[2]) == "sessions"
                and str(row[3]) == "session_id"
                and str(row[4]) == "session_id"
                and str(row[6]).upper() == "CASCADE"
                for row in foreign
            ):
                raise ValueError("SQLite journal schema v3 is missing a Session foreign key")
