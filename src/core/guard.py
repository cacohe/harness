from src.core.task import Task


class GuardError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HarnessGuard:
    def guard_input(self, task: Task) -> None: ...
