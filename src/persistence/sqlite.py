from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path

_SCHEMA_VERSION = 1
_CREATE_TASKS = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    snapshot TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout * 1000))}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                existing_tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                }
                if existing_tables:
                    raise TaskRepositoryError(
                        "unversioned database already contains application tables"
                    )
                connection.execute(_CREATE_TASKS)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise TaskRepositoryError(
                    f"unsupported schema version {version}; expected {_SCHEMA_VERSION}"
                )
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
