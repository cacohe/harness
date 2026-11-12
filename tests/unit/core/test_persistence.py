import importlib
import importlib.util

import pytest
from pydantic_core import PydanticSerializationError
from src.core import persistence
from src.core.task import Task
from src.schema.tool import ToolExecutionResult, ToolExecutionState


def test_core_exposes_task_repository_port():
    assert hasattr(persistence, "TaskRepository")


def test_in_memory_repository_saves_and_retrieves_tasks():
    try:
        spec = importlib.util.find_spec("src.persistence.in_memory")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None
    module = importlib.import_module("src.persistence.in_memory")
    repository = module.InMemoryTaskRepository()
    task = Task("persist me")

    repository.save(task)

    restored = repository.get(task.id)
    assert restored == task
    assert restored is not task
    assert repository.get("missing") is None


def test_in_memory_repository_isolates_saved_and_loaded_tasks():
    repository = importlib.import_module("src.persistence.in_memory").InMemoryTaskRepository()
    task = Task("original")
    repository.save(task)

    task.original_input = "changed after save"
    first_read = repository.get(task.id)
    assert first_read is not None
    assert first_read.original_input == "original"

    first_read.original_input = "changed after load"
    second_read = repository.get(task.id)
    assert second_read is not None
    assert second_read.original_input == "original"


def test_failed_snapshot_serialization_does_not_replace_last_valid_task():
    repository = importlib.import_module("src.persistence.in_memory").InMemoryTaskRepository()
    task = Task("valid")
    repository.save(task)
    task.pending_tool_calls = [
        ToolExecutionResult(
            tool_id="call-1",
            tool_name="broken",
            state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
            output=object(),
        )
    ]

    with pytest.raises(PydanticSerializationError):
        repository.save(task)

    restored = repository.get(task.id)
    assert restored is not None
    assert restored.original_input == "valid"
    assert restored.pending_tool_calls == []
