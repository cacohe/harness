from src.core import persistence


def test_core_exposes_task_repository_port():
    assert hasattr(persistence, "TaskRepository")
