import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from src.persistence import SQLiteTaskRepository, TaskRepositoryError


def _database_version(path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _table_names(path) -> set[str]:
    with sqlite3.connect(path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def test_constructor_creates_parent_version_and_tasks_table(tmp_path):
    path = tmp_path / "nested" / "tasks.db"

    repository = SQLiteTaskRepository(path)

    assert repository.path == path.resolve()
    assert path.is_file()
    assert _database_version(path) == 1
    assert _table_names(path) == {"tasks"}


def test_constructor_is_idempotent_for_an_existing_version_one_database(tmp_path):
    path = tmp_path / "tasks.db"

    first = SQLiteTaskRepository(path)
    second = SQLiteTaskRepository(path)

    assert first.path == second.path
    assert _database_version(path) == 1
    assert _table_names(path) == {"tasks"}


def test_concurrent_first_time_instances_initialize_one_database(tmp_path):
    path = tmp_path / "tasks.db"

    with ThreadPoolExecutor(max_workers=8) as pool:
        repositories = list(pool.map(lambda _index: SQLiteTaskRepository(path), range(16)))

    assert {repository.path for repository in repositories} == {path.resolve()}
    assert _database_version(path) == 1
    assert _table_names(path) == {"tasks"}


@pytest.mark.parametrize("path", [":memory:", "file::memory:?cache=shared"])
def test_constructor_rejects_memory_databases(path):
    with pytest.raises(TaskRepositoryError, match="filesystem database"):
        SQLiteTaskRepository(path)


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, True, "5"])
def test_constructor_rejects_invalid_timeout(tmp_path, timeout):
    with pytest.raises(TaskRepositoryError, match="timeout"):
        SQLiteTaskRepository(tmp_path / "tasks.db", timeout=timeout)


def test_constructor_wraps_invalid_database_path_without_leaking_it(tmp_path):
    path = tmp_path / "database-directory"
    path.mkdir()

    with pytest.raises(TaskRepositoryError, match="initialize") as caught:
        SQLiteTaskRepository(path)

    assert str(path) not in str(caught.value)


@pytest.mark.parametrize("version", [2, 99])
def test_constructor_rejects_unsupported_version_without_modifying_database(tmp_path, version):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
        connection.execute("CREATE TABLE future_data (value TEXT)")

    with pytest.raises(TaskRepositoryError, match="unsupported schema version") as caught:
        SQLiteTaskRepository(path)

    assert str(path) not in str(caught.value)
    assert _database_version(path) == version
    assert _table_names(path) == {"future_data"}


def test_persistence_package_exports_sqlite_repository():
    assert SQLiteTaskRepository.__name__ == "SQLiteTaskRepository"
    assert issubclass(TaskRepositoryError, RuntimeError)
