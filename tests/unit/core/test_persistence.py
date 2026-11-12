import importlib
import importlib.util

from src.core import persistence
from src.core.task import Task


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

    assert repository.get(task.id) is task
    assert repository.get("missing") is None
