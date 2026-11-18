from __future__ import annotations

import asyncio
import inspect
import json
import secrets
from dataclasses import dataclass, field
from typing import Callable

from src.adapters.tools.catalog import ToolCatalog, ToolCatalogEntry
from src.adapters.tools.execution_store import ToolExecutionStore
from src.adapters.tools.local import LocalToolGateway
from src.core.artifacts import ArtifactExternalizer
from src.core.tools import ToolSpec


@dataclass(slots=True)
class _Server:
    client: object
    generation: str
    entries: tuple[ToolCatalogEntry, ...]
    active: int = 0
    closing: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    closed: bool = False
    close_task: asyncio.Task[None] | None = None


class MCPToolGateway(LocalToolGateway):
    """Atomic generation-bound MCP catalog plus the durable local gateway contract."""

    def __init__(
        self,
        store: ToolExecutionStore,
        externalizer: ArtifactExternalizer,
        *,
        client_factory: Callable[[object], object],
        timeout_seconds: float = 60.0,
        workspace_id: str = "mcp",
    ) -> None:
        if not callable(client_factory) or type(workspace_id) is not str or not workspace_id:
            raise ValueError("MCP configuration is invalid")
        self._dynamic_catalog = ToolCatalog(())
        super().__init__(
            self._dynamic_catalog, store, externalizer, timeout_seconds=timeout_seconds
        )
        self._factory = client_factory
        self._workspace = workspace_id
        self._servers: dict[str, _Server] = {}
        self._lifecycle = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._pending_adds = 0
        self._adds_idle = asyncio.Event()
        self._adds_idle.set()

    async def add_server(self, name: str, source: object) -> None:
        _server_name(name)
        async with self._lifecycle:
            if self._closed:
                raise ValueError("MCP server cannot be added")
            self._pending_adds += 1
            self._adds_idle.clear()
        client: object | None = None
        try:
            client = await _invoke(
                self._factory,
                source,
                cancelled_result_cleanup=_close_once,
            )
            descriptors = await _await_method(client, "list_tools")
            generation = secrets.token_hex(16)
            entries = self._entries(name, generation, client, descriptors)
            async with self._lifecycle:
                if self._closed or name in self._servers:
                    raise ValueError("MCP server cannot be added")
                replacement = {**self._servers, name: _Server(client, generation, entries)}
                self._publish(replacement)
                self._servers = replacement
        except BaseException:
            if client is not None:
                await _owned_close(client)
            raise
        finally:
            async with self._lifecycle:
                self._pending_adds -= 1
                if self._pending_adds == 0:
                    self._adds_idle.set()

    async def refresh_server(self, name: str) -> None:
        async with self._lifecycle:
            state = self._servers.get(name)
            if state is None or state.closing:
                raise KeyError("MCP server is unavailable")
            state.closing = True
            async with state.condition:
                state.active += 1
        committed = False
        try:
            await self._wait_active(state, 1)
            descriptors = await _await_method(state.client, "list_tools")
            generation = secrets.token_hex(16)
            entries = self._entries(name, generation, state.client, descriptors)
            async with self._lifecycle:
                current = self._servers.get(name)
                if current is not state or self._closed:
                    raise RuntimeError("MCP server changed during refresh")
                replacement = dict(self._servers)
                replacement[name] = _Server(state.client, generation, entries)
                self._publish(replacement)
                self._servers = replacement
                committed = True
        finally:
            async with state.condition:
                state.active -= 1
                state.condition.notify_all()
            async with self._lifecycle:
                if not committed and self._servers.get(name) is state:
                    state.closing = False

    async def remove_server(self, name: str) -> None:
        async with self._lifecycle:
            state = self._servers.get(name)
            if state is None or state.closing:
                raise KeyError("MCP server is unavailable")
            state.closing = True
        committed = False
        try:
            await self._wait_idle(state)
            async with self._lifecycle:
                if self._servers.get(name) is not state:
                    raise RuntimeError("MCP server changed during removal")
                replacement = dict(self._servers)
                del replacement[name]
                self._publish(replacement)
                self._servers = replacement
                committed = True
            await self._shielded_close(state)
        finally:
            async with self._lifecycle:
                if not committed and self._servers.get(name) is state:
                    state.closing = False

    async def aclose(self) -> None:
        async with self._lifecycle:
            task = self._close_task
            if task is None:
                self._closed = True
                states = tuple(self._servers.values())
                for state in states:
                    state.closing = True
                self._servers = {}
                self._dynamic_catalog.replace(())
                task = asyncio.create_task(self._close_all(states))
                self._close_task = task
        await asyncio.shield(task)

    async def _close_all(self, states: tuple[_Server, ...]) -> None:
        await self._adds_idle.wait()
        failure: BaseException | None = None
        for state in states:
            try:
                await self._wait_idle(state)
                await self._shielded_close(state)
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def _entries(
        self, name: str, generation: str, client: object, descriptors: object
    ) -> tuple[ToolCatalogEntry, ...]:
        if type(descriptors) is not list or len(descriptors) > 128:
            raise ValueError("MCP tool catalog is invalid")
        entries: list[ToolCatalogEntry] = []
        for descriptor in descriptors:
            remote_name = _value(descriptor, "name")
            description = _value(descriptor, "description", "")
            schema = _value(descriptor, "inputSchema", _value(descriptor, "input_schema", {}))
            if (
                type(remote_name) is not str
                or type(description) is not str
                or type(schema) is not dict
            ):
                raise ValueError("MCP tool descriptor is invalid")
            public_name = f"{name}.{remote_name}"
            annotations = _value(descriptor, "annotations", {})
            mutating = True
            if type(annotations) is dict and annotations.get("readOnlyHint") is True:
                mutating = False
            workspace = f"{self._workspace}-{generation}"
            spec = ToolSpec(
                public_name,
                description,
                json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                requires_approval=mutating,
                approval_scope="execute",
                workspace_id=workspace,
            )

            async def call(
                arguments: dict[str, object],
                *,
                server_name: str = name,
                expected_generation: str = generation,
                tool_name: str = remote_name,
            ) -> object:
                return await self._call(server_name, expected_generation, tool_name, arguments)

            entries.append(ToolCatalogEntry(spec, call, mutating, generation))
        if len({item.spec.name for item in entries}) != len(entries):
            raise ValueError("MCP tool names are not unique")
        return tuple(entries)

    async def _call(
        self, server_name: str, generation: str, tool_name: str, arguments: dict[str, object]
    ) -> object:
        async with self._lifecycle:
            state = self._servers.get(server_name)
            if state is None or state.closing or state.generation != generation:
                raise RuntimeError("MCP generation is stale")
            async with state.condition:
                state.active += 1
        try:
            return await _await_method(state.client, "call_tool", tool_name, arguments)
        finally:
            async with state.condition:
                state.active -= 1
                state.condition.notify_all()

    async def _wait_idle(self, state: _Server) -> None:
        await self._wait_active(state, 0)

    async def _wait_active(self, state: _Server, expected: int) -> None:
        async with state.condition:
            await state.condition.wait_for(lambda: state.active == expected)

    async def _shielded_close(self, state: _Server) -> None:
        task = state.close_task
        if task is None:
            state.closed = True
            task = asyncio.create_task(_close_once(state.client))
            state.close_task = task
        await asyncio.shield(task)

    def _publish(self, servers: dict[str, _Server]) -> None:
        entries = tuple(entry for state in servers.values() for entry in state.entries)
        self._dynamic_catalog.replace(entries)


def _server_name(name: object) -> None:
    if type(name) is not str or not name.strip() or any(item in name for item in "/\\\x00"):
        raise ValueError("MCP server name is invalid")


def _value(value: object, name: str, default: object = ...) -> object:
    if type(value) is dict:
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)


async def _await_method(target: object, name: str, *args: object) -> object:
    method = getattr(target, name, None)
    if not callable(method):
        raise ValueError("MCP client contract is invalid")
    return await _invoke(method, *args)


async def _invoke(
    method: Callable[..., object],
    *args: object,
    cancelled_result_cleanup: Callable[[object], object] | None = None,
) -> object:
    if inspect.iscoroutinefunction(method):
        return await method(*args)
    result, cancellation = await _settle(asyncio.to_thread(method, *args))
    if inspect.isawaitable(result):
        result, cancellation = await _settle(result, cancellation)
    if cancellation is not None:
        if cancelled_result_cleanup is not None:
            try:
                cleanup = cancelled_result_cleanup(result)
                if inspect.isawaitable(cleanup):
                    await cleanup
            except BaseException:
                pass
        raise cancellation
    return result


async def _settle(
    awaitable: object,
    cancellation: asyncio.CancelledError | None = None,
) -> tuple[object, asyncio.CancelledError | None]:
    if not inspect.isawaitable(awaitable):
        raise TypeError("MCP awaitable contract is invalid")
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
    try:
        return task.result(), cancellation
    except BaseException:
        if cancellation is not None:
            raise cancellation
        raise


async def _close_once(client: object) -> None:
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if callable(closer):
        await _invoke(closer)


async def _owned_close(client: object) -> None:
    task = asyncio.create_task(_close_once(client))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
    await task
    if cancellation is not None:
        raise cancellation
