from typing import Protocol, runtime_checkable

from src.core.task import Task


class GuardError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@runtime_checkable
class HarnessGuard(Protocol):
    def guard_input(self, task: Task) -> None: ...


class NoOpGuard:
    """Allow every input while satisfying the guard port."""

    def guard_input(self, task: Task) -> None:
        return None
