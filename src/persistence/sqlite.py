from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path

from src.core.persistence import (
    ExecutionCommitError,
    ExecutionRecord,
    SequenceConflictError,
    SubmissionConflictError,
    _assign_events,
    _commit_session_id,
    _decode_event,
    _decode_submission,
    _decode_task,
    _encode_event,
    _encode_submission,
    _encode_task,
    _validate_expected_sequence,
    _validate_submission_update,
)
from src.core.task import Task
from src.schema.execution import PendingSessionEvent, SessionEvent, SubmissionRecord

_SCHEMA_VERSION = 2
_CREATE_TASKS = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    snapshot TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
_CREATE_SESSION_STREAMS = """
CREATE TABLE session_streams (
    session_id TEXT PRIMARY KEY,
    next_sequence INTEGER NOT NULL CHECK (next_sequence >= 1)
)
"""
_CREATE_SESSION_EVENTS = """
CREATE TABLE session_events (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
)
"""
_CREATE_SUBMISSIONS = """
CREATE TABLE submissions (
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    digest_algorithm TEXT NOT NULL,
    message_digest TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, message_id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
)
"""

_TASK_COLUMNS = [
    ("id", "TEXT", False, None, 1),
    ("snapshot", "TEXT", True, None, 0),
    ("updated_at", "TEXT", True, "CURRENT_TIMESTAMP", 0),
]
_STREAM_COLUMNS = [
    ("session_id", "TEXT", False, None, 1),
    ("next_sequence", "INTEGER", True, None, 0),
]
_EVENT_COLUMNS = [
    ("session_id", "TEXT", True, None, 1),
    ("sequence", "INTEGER", True, None, 2),
    ("event_id", "TEXT", True, None, 0),
    ("task_id", "TEXT", True, None, 0),
    ("event_type", "TEXT", True, None, 0),
    ("payload", "TEXT", True, None, 0),
    ("occurred_at", "TEXT", True, None, 0),
]
_SUBMISSION_COLUMNS = [
    ("session_id", "TEXT", True, None, 1),
    ("message_id", "TEXT", True, None, 2),
    ("digest_algorithm", "TEXT", True, None, 0),
    ("message_digest", "TEXT", True, None, 0),
    ("task_id", "TEXT", True, None, 0),
    ("turn_id", "TEXT", True, None, 0),
    ("state", "TEXT", True, None, 0),
    ("created_at", "TEXT", True, None, 0),
    ("updated_at", "TEXT", True, None, 0),
]
_V2_TABLES = {"tasks", "session_streams", "session_events", "submissions"}


class TaskRepositoryError(RuntimeError):
    """A durable task repository operation failed safely."""


class SQLiteTaskRepository:
    """Persist task snapshots and ordered execution facts in SQLite."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        raw_path = os.fspath(path)
        if raw_path == ":memory:" or raw_path.startswith("file::memory:"):
            raise TaskRepositoryError("task repository requires a filesystem database")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise TaskRepositoryError("task repository timeout must be a positive finite number")
        try:
            resolved = Path(raw_path).expanduser().resolve(strict=False)
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            raise TaskRepositoryError("could not initialize task repository") from error
        self.path = resolved
        self.timeout = float(timeout)
        self._initialize()

    def save(self, task: Task) -> None:
        """Atomically insert or replace one validated task snapshot."""
        try:
            encoded = _encode_task(task)
        except Exception as error:
            raise TaskRepositoryError("could not encode task snapshot") from error
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            self._upsert_task(connection, task.id, encoded)
            connection.commit()
        except Exception as error:
            if connection is not None:
                connection.rollback()
            raise TaskRepositoryError("could not save task snapshot") from error
        finally:
            if connection is not None:
                connection.close()

    def get(self, task_id: str) -> Task | None:
        """Load one independent task snapshot by id."""
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            row = connection.execute(
                "SELECT snapshot FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        except Exception as error:
            raise TaskRepositoryError("could not read task snapshot") from error
        finally:
            if connection is not None:
                connection.close()
        if row is None:
            return None
        try:
            task = _decode_task(str(row[0]))
            task.to_dict()
            return task
        except Exception as error:
            raise TaskRepositoryError("stored task snapshot is invalid") from error

    def commit(
        self,
        task: Task,
        events: Sequence[PendingSessionEvent],
        *,
        expected_sequence: int | None,
        submission: SubmissionRecord | None = None,
    ) -> list[SessionEvent]:
        """Atomically commit a snapshot, ordered facts, and submission state."""
        expected = _validate_expected_sequence(expected_sequence, allow_none=False)
        assert expected is not None
        pending = [PendingSessionEvent.model_validate(event.model_dump()) for event in events]
        incoming = None if submission is None else _decode_submission(_encode_submission(submission))
        session_id = _commit_session_id(task, pending, incoming)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT next_sequence FROM session_streams WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current = 0 if row is None else int(row[0]) - 1
            if current != expected:
                raise SequenceConflictError(
                    f"expected session sequence {expected}, found {current}"
                )
            existing = None
            if incoming is not None:
                existing = self._read_submission(
                    connection, incoming.session_id, incoming.message_id
                )
            incoming = _validate_submission_update(existing, incoming)
            assigned = _assign_events(pending, current)
            snapshot = Task.from_dict(task.to_dict())
            snapshot.last_sequence = current + len(assigned)
            encoded_task = _encode_task(snapshot)
            encoded_events = [(event, _encode_event(event)) for event in assigned]

            self._upsert_task(connection, snapshot.id, encoded_task)
            next_sequence = current + len(assigned) + 1
            if row is None:
                connection.execute(
                    "INSERT INTO session_streams (session_id, next_sequence) VALUES (?, ?)",
                    (session_id, next_sequence),
                )
            else:
                updated = connection.execute(
                    """
                    UPDATE session_streams SET next_sequence = ?
                    WHERE session_id = ? AND next_sequence = ?
                    """,
                    (next_sequence, session_id, current + 1),
                )
                if updated.rowcount != 1:
                    raise SequenceConflictError("session sequence changed during commit")
            connection.executemany(
                """
                INSERT INTO session_events
                    (session_id, sequence, event_id, task_id, event_type, payload, occurred_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.session_id,
                        event.sequence,
                        event.event_id,
                        event.task_id,
                        event.type.value,
                        encoded,
                        event.occurred_at.isoformat(),
                    )
                    for event, encoded in encoded_events
                ],
            )
            if incoming is not None:
                connection.execute(
                    """
                    INSERT INTO submissions
                        (session_id, message_id, digest_algorithm, message_digest, task_id,
                         turn_id, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, message_id) DO UPDATE SET
                        state = excluded.state,
                        updated_at = excluded.updated_at
                    """,
                    (
                        incoming.session_id,
                        incoming.message_id,
                        incoming.digest_algorithm,
                        incoming.message_digest,
                        incoming.task_id,
                        incoming.turn_id,
                        incoming.state.value,
                        incoming.created_at.isoformat(),
                        incoming.updated_at.isoformat(),
                    ),
                )
            connection.commit()
            return [_decode_event(encoded) for _, encoded in encoded_events]
        except (SequenceConflictError, SubmissionConflictError):
            if connection is not None:
                connection.rollback()
            raise
        except Exception as error:
            if connection is not None:
                connection.rollback()
            raise ExecutionCommitError("could not commit execution facts") from error
        finally:
            if connection is not None:
                connection.close()

    def events(self, session_id: str, *, after_sequence: int = 0) -> list[SessionEvent]:
        after = _validate_expected_sequence(after_sequence, allow_none=False)
        assert after is not None
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            rows = connection.execute(
                """
                SELECT payload FROM session_events
                WHERE session_id = ? AND sequence > ? ORDER BY sequence
                """,
                (session_id, after),
            ).fetchall()
            return [_decode_event(str(row[0])) for row in rows]
        except Exception as error:
            raise ExecutionCommitError("stored execution event is invalid") from error
        finally:
            if connection is not None:
                connection.close()

    def submission(self, session_id: str, message_id: str) -> SubmissionRecord | None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            return self._read_submission(connection, session_id, message_id)
        except Exception as error:
            raise ExecutionCommitError("stored submission is invalid") from error
        finally:
            if connection is not None:
                connection.close()

    def unfinished(self, session_id: str | None = None) -> list[ExecutionRecord]:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            rows = connection.execute("SELECT snapshot FROM tasks ORDER BY id").fetchall()
            records: list[ExecutionRecord] = []
            for row in rows:
                task = _decode_task(str(row[0]))
                if task.is_finished() or task.session_id is None:
                    continue
                if session_id is not None and task.session_id != session_id:
                    continue
                records.append(ExecutionRecord(task.session_id, task, task.last_sequence or 0))
            return records
        except Exception as error:
            raise ExecutionCommitError("stored task snapshot is invalid") from error
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _upsert_task(connection: sqlite3.Connection, task_id: str, encoded: str) -> None:
        connection.execute(
            """
            INSERT INTO tasks (id, snapshot, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                snapshot = excluded.snapshot,
                updated_at = CURRENT_TIMESTAMP
            """,
            (task_id, encoded),
        )

    @staticmethod
    def _read_submission(
        connection: sqlite3.Connection, session_id: str, message_id: str
    ) -> SubmissionRecord | None:
        row = connection.execute(
            """
            SELECT session_id, message_id, digest_algorithm, message_digest, task_id,
                   turn_id, state, created_at, updated_at
            FROM submissions WHERE session_id = ? AND message_id = ?
            """,
            (session_id, message_id),
        ).fetchone()
        if row is None:
            return None
        names = (
            "session_id", "message_id", "digest_algorithm", "message_digest", "task_id",
            "turn_id", "state", "created_at", "updated_at",
        )
        encoded = json.dumps(
            dict(zip(names, row, strict=True)),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return _decode_submission(encoded)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        try:
            connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout * 1000))}")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _application_tables(connection: sqlite3.Connection) -> set[str]:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
        return [
            (str(row[1]), str(row[2]).upper(), bool(row[3]), row[4], int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        ]

    @classmethod
    def _validate_v1_schema(cls, connection: sqlite3.Connection) -> None:
        if (
            cls._application_tables(connection) != {"tasks"}
            or cls._columns(connection, "tasks") != _TASK_COLUMNS
        ):
            raise TaskRepositoryError("task repository schema does not match version 1")

    @classmethod
    def _validate_v2_schema(cls, connection: sqlite3.Connection) -> None:
        expected = {
            "tasks": _TASK_COLUMNS,
            "session_streams": _STREAM_COLUMNS,
            "session_events": _EVENT_COLUMNS,
            "submissions": _SUBMISSION_COLUMNS,
        }
        if cls._application_tables(connection) != _V2_TABLES or any(
            cls._columns(connection, table) != columns for table, columns in expected.items()
        ):
            raise TaskRepositoryError("task repository schema does not match version 2")
        for table in ("session_events", "submissions"):
            actual = [
                (str(row[2]), str(row[3]), str(row[4]))
                for row in connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
            ]
            if actual != [("tasks", "task_id", "id")]:
                raise TaskRepositoryError("task repository schema does not match version 2")

    @classmethod
    def _validate_compatible_database(cls, connection: sqlite3.Connection) -> int:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            if cls._application_tables(connection):
                raise TaskRepositoryError(
                    "unversioned database already contains application tables"
                )
        elif version == 1:
            cls._validate_v1_schema(connection)
        elif version == _SCHEMA_VERSION:
            cls._validate_v2_schema(connection)
        else:
            raise TaskRepositoryError(
                f"unsupported schema version {version}; expected {_SCHEMA_VERSION}"
            )
        return version

    def _enable_wal(self, connection: sqlite3.Connection) -> None:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                current = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
                if current.lower() == "wal":
                    return
                selected = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
                if selected.lower() != "wal":
                    raise sqlite3.OperationalError("could not enable WAL journal mode")
                return
            except sqlite3.OperationalError as error:
                if "locked" not in str(error).lower() or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def _initialize(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN")
            try:
                self._validate_compatible_database(connection)
            finally:
                connection.rollback()
            self._enable_wal(connection)
            connection.execute("BEGIN IMMEDIATE")
            version = self._validate_compatible_database(connection)
            if version == 0:
                connection.execute(_CREATE_TASKS)
            if version in {0, 1}:
                connection.execute(_CREATE_SESSION_STREAMS)
                connection.execute(_CREATE_SESSION_EVENTS)
                connection.execute(_CREATE_SUBMISSIONS)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                self._validate_v2_schema(connection)
            connection.commit()
        except TaskRepositoryError:
            if connection is not None:
                connection.rollback()
            raise
        except (sqlite3.Error, OSError, RuntimeError, ValueError, TypeError) as error:
            if connection is not None:
                connection.rollback()
            raise TaskRepositoryError("could not initialize task repository") from error
        finally:
            if connection is not None:
                connection.close()
