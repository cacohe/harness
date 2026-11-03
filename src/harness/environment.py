from __future__ import annotations

from harness.types import EnvHandle, ToolCall, ToolResult


class CurrentEnvironment:
    kind = "current"

    def __init__(self, executor) -> None:
        self.executor = executor

    def prepare(self) -> EnvHandle:
        return EnvHandle(kind=self.kind, id="current")

    def invoke(self, handle: EnvHandle, call: ToolCall) -> ToolResult:
        return self.executor.execute(call, handle)

    def dispose(self, handle: EnvHandle) -> None:
        return None


class IsolatedEnvironment:
    """Isolation is a port. Creating and calling a sandbox is left to the host."""

    kind = "isolated"

    def prepare(self) -> EnvHandle:
        raise NotImplementedError("TODO: generate isolated environment")

    def invoke(self, handle: EnvHandle, call: ToolCall) -> ToolResult:
        raise NotImplementedError("TODO: invoke isolated environment")

    def dispose(self, handle: EnvHandle) -> None:
        return None
