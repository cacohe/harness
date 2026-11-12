import json
from typing import Any

from src.core.task import Task


class InMemoryTaskRepository:
    """Keep task objects for the lifetime of one process."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}

    def save(self, task: Task) -> None:
        snapshot = task.to_dict()
        encoded = json.dumps(snapshot)
        self._tasks[task.id] = json.loads(encoded)

    def get(self, task_id: str) -> Task | None:
        snapshot = self._tasks.get(task_id)
        if snapshot is None:
            return None
        return Task.from_dict(snapshot)
