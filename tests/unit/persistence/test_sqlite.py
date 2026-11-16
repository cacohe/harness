import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread

import pytest
import src.persistence.sqlite as sqlite_module
from src.core.persistence import ExecutionCommitError, SequenceConflictError
from src.core.task import Task
from src.persistence import SQLiteTaskRepository, TaskRepositoryError
from src.schema.context import Message
from src.schema.execution import (
    PendingSessionEvent,
    SessionEventType,
    SubmissionRecord,
    SubmissionState,
)
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


def _create_v1_database(path, tasks=()):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                snapshot TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        for task in tasks:
            connection.execute(
                "INSERT INTO tasks (id, snapshot) VALUES (?, ?)",
                (task.id, json.dumps(task.to_dict(), ensure_ascii=False)),
            )
        connection.execute("PRAGMA user_version = 1")


def _execution_event(event_id="event-1"):
    return PendingSessionEvent(
        event_id=event_id,
        session_id="session-1",
        type=SessionEventType.TURN_STARTED,
        task_id="task-1",
        execution_id="execution-1",
        turn_id="turn-1",
        payload={},
    )


def _execution_task(text="work"):
    return Task(
        text,
        id="task-1",
        session_id="session-1",
        message_id="message-1",
        turn_id="turn-1",
    )


def _submission(digest="a" * 64):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return SubmissionRecord(
        session_id="session-1",
        message_id="message-1",
        digest_algorithm="sha256-raw-utf8-v1",
        message_digest=digest,
        task_id="task-1",
        turn_id="turn-1",
        state=SubmissionState.ACCEPTED,
        created_at=now,
        updated_at=now,
    )


def test_constructor_creates_parent_version_and_v2_tables(tmp_path):
    path = tmp_path / "nested" / "tasks.db"

    repository = SQLiteTaskRepository(path)

    assert repository.path == path.resolve()
    assert path.is_file()
    assert _database_version(path) == 2
    assert _table_names(path) == {
        "tasks",
        "session_streams",
        "session_events",
        "submissions",
    }


def test_constructor_is_idempotent_for_an_existing_version_two_database(tmp_path):
    path = tmp_path / "tasks.db"

    first = SQLiteTaskRepository(path)
    second = SQLiteTaskRepository(path)

    assert first.path == second.path
    assert _database_version(path) == 2
    assert _table_names(path) == {
        "tasks",
        "session_streams",
        "session_events",
        "submissions",
    }


def test_concurrent_first_time_instances_initialize_one_database(tmp_path):
    path = tmp_path / "tasks.db"

    with ThreadPoolExecutor(max_workers=8) as pool:
        repositories = list(pool.map(lambda _index: SQLiteTaskRepository(path), range(16)))

    assert {repository.path for repository in repositories} == {path.resolve()}
    assert _database_version(path) == 2
    assert _table_names(path) == {
        "tasks",
        "session_streams",
        "session_events",
        "submissions",
    }


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
    assert _database_version(path) == 2
    assert _table_names(path) == {
        "tasks",
        "session_streams",
        "session_events",
        "submissions",
    }


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


@pytest.mark.parametrize("version", [3, 99])
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


@pytest.mark.parametrize("tasks", [[], [Task("first", id="task-1"), Task("second", id="task-2")]])
def test_constructor_migrates_v1_tasks_to_v2_without_fabricating_events(tmp_path, tasks):
    path = tmp_path / "v1.db"
    _create_v1_database(path, tasks)

    repository = SQLiteTaskRepository(path)

    assert _database_version(path) == 2
    assert [repository.get(task.id) for task in tasks] == tasks
    assert repository.events("legacy") == []


def test_failed_v1_migration_rolls_back_every_schema_change(tmp_path, monkeypatch):
    path = tmp_path / "v1.db"
    _create_v1_database(path, [Task("keep", id="task-1")])
    monkeypatch.setattr(sqlite_module, "_CREATE_SUBMISSIONS", "CREATE TABL broken", raising=False)

    with pytest.raises(TaskRepositoryError, match="initialize"):
        SQLiteTaskRepository(path)

    assert _database_version(path) == 1
    assert _table_names(path) == {"tasks"}
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_constructor_rejects_version_two_with_missing_execution_tables(tmp_path):
    path = tmp_path / "invalid-v2.db"
    _create_v1_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 2")

    with pytest.raises(TaskRepositoryError, match="schema"):
        SQLiteTaskRepository(path)


def test_v2_schema_has_expected_foreign_keys(tmp_path):
    path = tmp_path / "tasks.db"
    SQLiteTaskRepository(path)

    with sqlite3.connect(path) as connection:
        event_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(session_events)"
        ).fetchall()
        submission_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(submissions)"
        ).fetchall()

    assert [(row[2], row[3], row[4]) for row in event_foreign_keys] == [("tasks", "task_id", "id")]
    assert [(row[2], row[3], row[4]) for row in submission_foreign_keys] == [
        ("tasks", "task_id", "id")
    ]


def test_events_reject_corrupt_stored_payload(tmp_path):
    path = tmp_path / "tasks.db"
    repository = SQLiteTaskRepository(path)
    repository.commit(_execution_task(), [_execution_event()], expected_sequence=0)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE session_events SET payload = ? WHERE event_id = ?",
            ("{not-json", "event-1"),
        )

    with pytest.raises(ExecutionCommitError, match="stored execution event is invalid"):
        repository.events("session-1")


def test_execution_commit_reopens_with_snapshot_events_and_submission(tmp_path):
    path = tmp_path / "tasks.db"
    SQLiteTaskRepository(path).commit(
        _execution_task(),
        [_execution_event()],
        expected_sequence=0,
        submission=_submission(),
    )

    reopened = SQLiteTaskRepository(path)

    assert reopened.get("task-1").last_sequence == 1
    assert [event.event_id for event in reopened.events("session-1")] == ["event-1"]
    assert reopened.submission("session-1", "message-1").message_digest == "a" * 64


def test_two_sqlite_writers_cannot_commit_the_same_expected_sequence(tmp_path):
    path = tmp_path / "tasks.db"
    first = SQLiteTaskRepository(path, timeout=10)
    second = SQLiteTaskRepository(path, timeout=10)

    def commit(repository, event_id):
        return repository.commit(
            _execution_task(event_id),
            [_execution_event(event_id)],
            expected_sequence=0,
            submission=_submission(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(commit, first, "event-1"), pool.submit(commit, second, "event-2")]
    results = []
    errors = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception as error:
            errors.append(error)

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], SequenceConflictError)
    stored = SQLiteTaskRepository(path)
    assert len(stored.events("session-1")) == 1
    assert stored.get("task-1").original_input in {"event-1", "event-2"}
    assert stored.submission("session-1", "message-1") is not None


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
