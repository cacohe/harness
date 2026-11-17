from typing import Protocol, runtime_checkable
from uuid import uuid4

from src.schema.execution import validate_execution_id as validate_execution_id


@runtime_checkable
class ExecutionIdFactory(Protocol):
    def session(self) -> str: ...

    def execution(self) -> str: ...

    def turn(self) -> str: ...

    def step(self) -> str: ...

    def attempt(self) -> str: ...

    def tool_call(self) -> str: ...

    def event(self) -> str: ...

    def message(self) -> str: ...


class UuidExecutionIds:
    """Generate opaque UUID identities for every execution scope."""

    @staticmethod
    def _new() -> str:
        return str(uuid4())

    def session(self) -> str:
        return self._new()

    def execution(self) -> str:
        return self._new()

    def turn(self) -> str:
        return self._new()

    def step(self) -> str:
        return self._new()

    def attempt(self) -> str:
        return self._new()

    def tool_call(self) -> str:
        return self._new()

    def event(self) -> str:
        return self._new()

    def message(self) -> str:
        return self._new()
