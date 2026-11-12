from typing import Protocol, runtime_checkable

from src.core.task import Task


@runtime_checkable
class TaskRepository(Protocol):
    def save(self, task: Task) -> None: ...
    def get(self, task_id: str) -> Task | None: ...


class NoOpTaskRepository:
    """Explicitly disable task persistence."""

    def save(self, task: Task) -> None:
        return None

    def get(self, task_id: str) -> Task | None:
        return None
