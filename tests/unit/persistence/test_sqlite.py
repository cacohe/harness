import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pytest
import src.persistence.sqlite as sqlite_module
from src.core.task import Task
from src.persistence import SQLiteTaskRepository, TaskRepositoryError
from src.schema.context import Message
from src.schema.tool import ToolExecutionResult, ToolExecutionState


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


def test_concurrent_initialization_reads_version_and_tables_from_one_snapshot(
    tmp_path, monkeypatch
):
    path = tmp_path / "tasks.db"
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    version_read = Event()
    continue_table_read = Event()
    original = SQLiteTaskRepository._application_tables

    def delayed_table_read(connection):
        if current_thread().name.startswith("delayed-init"):
            version_read.set()
            assert continue_table_read.wait(5)
        return original(connection)

    monkeypatch.setattr(
        SQLiteTaskRepository, "_application_tables", staticmethod(delayed_table_read)
    )

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="delayed-init") as pool:
        delayed = pool.submit(SQLiteTaskRepository, path)
        assert version_read.wait(5)
        try:
            immediate = SQLiteTaskRepository(path)
        finally:
            continue_table_read.set()
        first = delayed.result(timeout=5)

    assert first.path == immediate.path
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
    before = path.read_bytes()

    with pytest.raises(TaskRepositoryError, match="unsupported schema version") as caught:
        SQLiteTaskRepository(path)

    assert path.read_bytes() == before
    assert str(path) not in str(caught.value)
    assert _database_version(path) == version
    assert _table_names(path) == {"future_data"}


@pytest.mark.parametrize(
    "tasks_ddl",
    [
        None,
        "CREATE TABLE tasks (id TEXT, snapshot TEXT NOT NULL, "
        "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
    ],
)
def test_constructor_rejects_version_one_with_an_invalid_tasks_schema(tmp_path, tasks_ddl):
    path = tmp_path / "invalid-v1.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 1")
        if tasks_ddl is not None:
            connection.execute(tasks_ddl)

    with pytest.raises(TaskRepositoryError, match="schema") as caught:
        SQLiteTaskRepository(path)

    assert str(path) not in str(caught.value)


def test_failed_connection_configuration_closes_the_opened_connection(tmp_path, monkeypatch):
    path = tmp_path / "corrupt.db"
    path.write_bytes(b"not a sqlite database")
    opened = []
    real_connect = sqlite3.connect

    def capturing_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite_module.sqlite3, "connect", capturing_connect)

    with pytest.raises(TaskRepositoryError, match="initialize"):
        SQLiteTaskRepository(path)

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_persistence_package_exports_sqlite_repository():
    assert SQLiteTaskRepository.__name__ == "SQLiteTaskRepository"
    assert issubclass(TaskRepositoryError, RuntimeError)


def test_get_returns_none_for_a_missing_task(tmp_path):
    repository = SQLiteTaskRepository(tmp_path / "tasks.db")

    assert repository.get("missing") is None


def test_save_round_trips_unicode_through_a_new_repository_instance(tmp_path):
    path = tmp_path / "tasks.db"
    task = Task(
        "研究并总结",
        history=[Message(role="assistant", content="证据 ✅")],
    )
    SQLiteTaskRepository(path).save(task)

    restored = SQLiteTaskRepository(path).get(task.id)

    assert restored == task
    assert restored is not task


def test_save_overwrites_and_loaded_objects_are_independent(tmp_path):
    repository = SQLiteTaskRepository(tmp_path / "tasks.db")
    task = Task("first")
    repository.save(task)
    task.original_input = "second"
    repository.save(task)
    task.original_input = "changed after save"

    first = repository.get(task.id)
    assert first is not None
    assert first.original_input == "second"
    first.original_input = "changed after load"

    second = repository.get(task.id)
    assert second is not None
    assert second.original_input == "second"


@pytest.mark.parametrize(
    "snapshot",
    [
        "{not-json",
        json.dumps(
            {
                "schema_version": 1,
                "id": "broken",
                "original_input": "broken",
                "state": "completed",
                "state_reason": None,
                "plan": None,
                "history": [],
                "pending_tool_calls": [],
                "result": None,
                "step_count": 0,
                "elapsed_seconds": 0.0,
            }
        ),
    ],
)
def test_get_wraps_corrupt_or_invariant_invalid_snapshots(tmp_path, snapshot):
    path = tmp_path / "tasks.db"
    repository = SQLiteTaskRepository(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO tasks (id, snapshot) VALUES (?, ?)",
            ("broken", snapshot),
        )

    with pytest.raises(TaskRepositoryError, match="stored task snapshot is invalid") as caught:
        repository.get("broken")

    assert str(path) not in str(caught.value)


def test_get_rejects_nonstandard_nonfinite_json_values(tmp_path):
    path = tmp_path / "tasks.db"
    repository = SQLiteTaskRepository(path)
    task = Task(
        "broken",
        id="broken",
        pending_tool_calls=[
            ToolExecutionResult(
                tool_id="call-1",
                tool_name="tool",
                state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
                output=0,
            )
        ],
    )
    snapshot = task.to_dict()
    snapshot["pending_tool_calls"][0]["output"] = math.nan
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO tasks (id, snapshot) VALUES (?, ?)",
            (task.id, json.dumps(snapshot)),
        )

    with pytest.raises(TaskRepositoryError, match="stored task snapshot is invalid"):
        repository.get(task.id)


def test_failed_upsert_rolls_back_and_preserves_previous_snapshot(tmp_path):
    path = tmp_path / "tasks.db"
    repository = SQLiteTaskRepository(path)
    task = Task("committed")
    repository.save(task)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_task_update
            BEFORE UPDATE ON tasks
            BEGIN
                SELECT RAISE(ABORT, 'blocked');
            END
            """
        )
    task.original_input = "must roll back"

    with pytest.raises(TaskRepositoryError, match="save task snapshot") as caught:
        repository.save(task)

    assert "blocked" not in str(caught.value)
    assert str(path) not in str(caught.value)
    restored = repository.get(task.id)
    assert restored is not None
    assert restored.original_input == "committed"


def test_invalid_task_snapshot_does_not_replace_previous_snapshot(tmp_path):
    repository = SQLiteTaskRepository(tmp_path / "tasks.db")
    task = Task("committed")
    repository.save(task)
    task.pending_tool_calls = [
        ToolExecutionResult(
            tool_id="call-1",
            tool_name="broken",
            state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
            output=object(),
        )
    ]

    with pytest.raises(TaskRepositoryError, match="encode task snapshot"):
        repository.save(task)

    restored = repository.get(task.id)
    assert restored is not None
    assert restored.pending_tool_calls == []


def test_multiple_instances_concurrently_save_distinct_tasks(tmp_path):
    path = tmp_path / "tasks.db"
    tasks = [Task(f"input-{index}", id=f"task-{index}") for index in range(40)]

    def save(task):
        SQLiteTaskRepository(path, timeout=10).save(task)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, tasks))

    repository = SQLiteTaskRepository(path)
    assert [repository.get(task.id) for task in tasks] == tasks


def test_concurrent_upserts_of_one_id_leave_one_complete_submitted_snapshot(tmp_path):
    path = tmp_path / "tasks.db"
    tasks = [
        Task(
            f"version-{index}",
            id="shared",
            history=[Message(role="assistant", content=f"history-{index}")],
        )
        for index in range(40)
    ]
    submitted = [task.to_dict() for task in tasks]

    def save(task):
        SQLiteTaskRepository(path, timeout=10).save(task)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, tasks))

    restored = SQLiteTaskRepository(path).get("shared")
    assert restored is not None
    assert restored.to_dict() in submitted
