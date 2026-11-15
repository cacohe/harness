from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from pathlib import Path

from src.core.task import Task

_SCHEMA_VERSION = 1
_CREATE_TASKS = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    snapshot TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
_TASK_COLUMNS = [
    ("id", "TEXT", False, None, 1),
    ("snapshot", "TEXT", True, None, 0),
    ("updated_at", "TEXT", True, "CURRENT_TIMESTAMP", 0),
]


class TaskRepositoryError(RuntimeError):
    """A durable task repository operation failed safely."""


class SQLiteTaskRepository:
    """Persist task snapshots in one versioned SQLite database file."""

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
            snapshot = task.to_dict()
            encoded = json.dumps(
                snapshot,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            Task.from_dict(json.loads(encoded))
        except Exception as error:
            raise TaskRepositoryError("could not encode task snapshot") from error

        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO tasks (id, snapshot, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    snapshot = excluded.snapshot,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (task.id, encoded),
            )
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
            decoded = json.loads(row[0])
            task = Task.from_dict(decoded)
            task.to_dict()
            return task
        except Exception as error:
            raise TaskRepositoryError("stored task snapshot is invalid") from error

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        try:
            connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout * 1000))}")
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
    def _validate_schema(connection: sqlite3.Connection) -> None:
        columns = [
            (
                str(row[1]),
                str(row[2]).upper(),
                bool(row[3]),
                row[4],
                int(row[5]),
            )
            for row in connection.execute("PRAGMA table_info(tasks)").fetchall()
        ]
        if columns != _TASK_COLUMNS:
            raise TaskRepositoryError("task repository schema does not match version 1")

    @classmethod
    def _validate_compatible_database(cls, connection: sqlite3.Connection) -> int:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            if cls._application_tables(connection):
                raise TaskRepositoryError(
                    "unversioned database already contains application tables"
                )
        elif version == _SCHEMA_VERSION:
            cls._validate_schema(connection)
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
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                self._validate_schema(connection)
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
