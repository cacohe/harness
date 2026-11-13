import json

from src.core.task import Task


class InMemoryTaskRepository:
    """Keep task objects for the lifetime of one process."""

    def __init__(self) -> None:
        self._tasks: dict[str, str] = {}

    def save(self, task: Task) -> None:
        snapshot = task.to_dict()
        encoded = json.dumps(snapshot, allow_nan=False)
        Task.from_dict(json.loads(encoded))
        self._tasks[task.id] = encoded

    def get(self, task_id: str) -> Task | None:
        encoded = self._tasks.get(task_id)
        if encoded is None:
            return None
        return Task.from_dict(json.loads(encoded))
