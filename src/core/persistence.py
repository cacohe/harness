from typing import Any, Protocol, runtime_checkable

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


class CheckpointStore(Protocol):
    def save(self, key: str, data: Any, metadata: dict | None = None) -> None: ...
    def load(self, key: str) -> Any | None: ...
    def delete(self, key: str) -> None: ...


class ArtifactStore(Protocol):
    def save(self, key: str, data: Any, metadata: dict | None = None) -> None: ...
    def load(self, key: str) -> Any | None: ...
    def delete(self, key: str) -> None: ...


class TraceStore(Protocol):
    def append(self, task_id: str, event: Any) -> None: ...
