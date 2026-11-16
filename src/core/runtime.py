from __future__ import annotations

import asyncio
import inspect
from typing import Protocol, runtime_checkable


class RuntimeClosedError(RuntimeError):
    """The runtime no longer accepts new operations."""


class RuntimeCloseTimeoutError(TimeoutError):
    """The caller's close deadline elapsed while shielded cleanup continues."""


@runtime_checkable
class RuntimeResource(Protocol):
    async def aclose(self) -> None: ...


class RuntimeCoordinator:
    """Start owned resources and settle active operations before reverse-order close."""

    def __init__(self, resources: list[object] | None = None) -> None:
        self.resources = tuple(resources or [])
        self.state = "new"
        self._started: list[object] = []
        self._operations: set[asyncio.Task] = set()
        self._start_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.state in {"closing", "closed"}:
            raise RuntimeClosedError("runtime is closed")
        if self.state == "started":
            return
        async with self._start_lock:
            if self.state == "started":
                return
            if self.state in {"closing", "closed"}:
                raise RuntimeClosedError("runtime is closed")
            try:
                for resource in self.resources:
                    await self._invoke(resource, "astart", "start")
                    self._started.append(resource)
            except BaseException:
                for resource in reversed(self._started):
                    await self._invoke(resource, "aclose", "close")
                self._started.clear()
                self.state = "closed"
                raise
            self.state = "started"

    def ensure_accepting(self) -> None:
        if self.state in {"closing", "closed"}:
            raise RuntimeClosedError("runtime is closed")

    def register_operation(self, task: asyncio.Task) -> asyncio.Task:
        self.ensure_accepting()
        self._operations.add(task)
        task.add_done_callback(self._operations.discard)
        return task

    async def aclose(self, timeout_seconds: float) -> None:
        caller = asyncio.current_task()
        if caller is not None:
            self._operations.discard(caller)
        if self._cleanup_task is None:
            if self.state == "new":
                self._started = list(self.resources)
            self.state = "closing"
            self._cleanup_task = asyncio.create_task(self._cleanup())
        try:
            async with asyncio.timeout(timeout_seconds):
                await asyncio.shield(self._cleanup_task)
        except TimeoutError as error:
            raise RuntimeCloseTimeoutError("runtime close timed out") from error

    async def _cleanup(self) -> None:
        current = asyncio.current_task()
        operations = [task for task in self._operations if task is not current and not task.done()]
        for task in operations:
            task.cancel()
        if operations:
            await asyncio.gather(*operations, return_exceptions=True)
        for resource in reversed(self._started):
            await self._invoke(resource, "aflush", "flush")
        for resource in reversed(self._started):
            await self._invoke(resource, "aclose", "close")
        self._started.clear()
        self.state = "closed"

    @staticmethod
    async def _invoke(resource: object, async_name: str, sync_name: str) -> None:
        method = getattr(resource, async_name, None) or getattr(resource, sync_name, None)
        if method is None:
            return
        result = method()
        if inspect.isawaitable(result):
            await result
