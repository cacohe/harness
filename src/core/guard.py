from src.core.task import Task


class GuardError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def guard(task: Task) -> None: ...
