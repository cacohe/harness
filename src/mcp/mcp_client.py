from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client

from src.core.tool import RemoteToolError, ToolCatalog
from src.mcp.tool_adapter import make_mcp_tool
from src.schema.tool import Tool


@dataclass
class _ServerState:
    client: Any
    stack: AsyncExitStack
    tools: dict[str, Tool]
    generation: str
    active_calls: int = 0
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    closing: bool = False


class MCPManager:
    """Own MCP transports and expose their tools through one catalog."""

    def __init__(
        self,
        catalog: ToolCatalog,
        *,
        client_factory: Callable[..., Any] = Client,
    ) -> None:
        self._catalog = catalog
        self._client_factory = client_factory
        self._servers: dict[str, _ServerState] = {}
        self._closing_names: set[str] = set()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def catalog(self) -> ToolCatalog:
        return self._catalog

    async def add_server(self, name: str, source: Any, **options: Any) -> None:
        """Connect a server and atomically publish all of its tools."""
        self._validate_name(name)
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MCP manager is closed")
            if name in self._servers or name in self._closing_names:
                raise ValueError(f"MCP server {name!r} is already connected")

            stack = AsyncExitStack()
            await stack.__aenter__()
            registration_started = False
            tools: dict[str, Tool] = {}
            try:
                candidate = self._client_factory(self._client_source(name, source), **options)
                stack.push_async_callback(self._close_client, candidate)
                client = await stack.enter_async_context(candidate)
                generation = uuid.uuid4().hex
                tools = self._build_tools(name, await client.list_tools(), generation)
                self._preflight_tools(tools.values())
                registration_started = True
                self._register_tools(tools.values())
                self._servers[name] = _ServerState(
                    client=client,
                    stack=stack,
                    tools=tools,
                    generation=generation,
                )
            except BaseException:
                if registration_started:
                    self._unregister_tools(tools)
                await stack.aclose()
                raise

    async def refresh_server(self, name: str) -> None:
        """Replace one server's advertised tools as an atomic catalog update."""
        async with self._lifecycle_lock:
            state = self._servers.get(name)
            if state is None or state.closing:
                raise KeyError(f"MCP server {name!r} is not connected")

            generation = uuid.uuid4().hex
            tools = self._build_tools(name, await state.client.list_tools(), generation)
            old_tools = list(state.tools.values())
            old_ids = {tool.id for tool in old_tools}
            self._preflight_tools(tools.values(), ignored_ids=old_ids)

            self._unregister_tools(old_ids)
            try:
                self._register_tools(tools.values())
            except BaseException as registration_error:
                self._unregister_tools(tools)
                try:
                    for tool in old_tools:
                        self._catalog.register(tool)
                except BaseException as restore_error:
                    raise BaseExceptionGroup(
                        "MCP refresh registration and rollback both failed",
                        [registration_error, restore_error],
                    ) from registration_error
                raise
            state.tools = tools
            state.generation = generation

    async def remove_server(self, name: str) -> None:
        """Unpublish a server and wait for its owned cleanup task."""
        async with self._lifecycle_lock:
            state = self._servers.pop(name, None)
            if state is None:
                raise KeyError(f"MCP server {name!r} is not connected")
            cleanup = self._start_cleanup_locked(name, state)
        await asyncio.shield(cleanup)

    async def aclose(self) -> None:
        """Atomically unpublish all servers, then drain every owned cleanup task."""
        async with self._lifecycle_lock:
            if self._close_task is None:
                self._closed = True
                for name, state in list(self._servers.items()):
                    self._servers.pop(name)
                    self._start_cleanup_locked(name, state)
                cleanups = list(self._cleanup_tasks)
                self._close_task = asyncio.create_task(
                    self._finish_close(cleanups), name="mcp-manager-close"
                )
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def __aenter__(self) -> MCPManager:
        if self._closed:
            raise RuntimeError("MCP manager is closed")
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()

    async def add_mcp_server(self, name: str, source: Any, **options: Any) -> None:
        """Compatibility alias for :meth:`add_server`."""
        await self.add_server(name, source, **options)

    async def remove_mcp_server(self, name: str) -> None:
        """Compatibility alias for :meth:`remove_server`."""
        await self.remove_server(name)

    async def list_tools(self) -> list[Tool]:
        """Return the currently published MCP tools."""
        async with self._lifecycle_lock:
            return [tool for state in self._servers.values() for tool in state.tools.values()]

    async def call_tool(self, tool: Tool, arguments: dict | None = None) -> Any:
        """Call a tool returned by this manager."""
        return await tool.executer(arguments)

    def _start_cleanup_locked(self, name: str, state: _ServerState) -> asyncio.Task[None]:
        state.closing = True
        self._closing_names.add(name)
        self._unregister_tools(state.tools)
        task = asyncio.create_task(self._drain_and_close(name, state), name=f"mcp-close-{name}")
        task.add_done_callback(self._consume_cleanup_result)
        self._cleanup_tasks.add(task)
        return task

    async def _drain_and_close(self, name: str, state: _ServerState) -> None:
        try:
            async with state.condition:
                await state.condition.wait_for(lambda: state.active_calls == 0)
            await state.stack.aclose()
        finally:
            async with self._lifecycle_lock:
                self._closing_names.discard(name)

    async def _finish_close(self, cleanups: list[asyncio.Task[None]]) -> None:
        results = await asyncio.gather(*cleanups, return_exceptions=True)
        async with self._lifecycle_lock:
            self._cleanup_tasks.difference_update(cleanups)
        errors = [result for result in results if isinstance(result, BaseException)]
        if not errors:
            return
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("failed to close MCP servers", errors)

    @staticmethod
    def _consume_cleanup_result(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    @staticmethod
    async def _close_client(client: Any) -> None:
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    def _build_tools(
        self, server_name: str, remote_tools: Iterable[Any], generation: str
    ) -> dict[str, Tool]:
        tools: dict[str, Tool] = {}
        invoke = self._make_invoker(server_name, generation)
        for remote_tool in remote_tools:
            tool = make_mcp_tool(server_name, remote_tool, invoke)
            tool.approval_scope = f"mcp:{generation}:{tool.id}"
            if tool.id in tools or any(existing.name == tool.name for existing in tools.values()):
                raise ValueError(f"tool collision for id {tool.id!r} or name {tool.name!r}")
            tools[tool.id] = tool
        return tools

    def _make_invoker(
        self, server_name: str, generation: str
    ) -> Callable[[str, dict[str, Any]], Any]:
        async def invoke(remote_name: str, arguments: dict[str, Any]) -> Any:
            return await self._invoke(server_name, generation, remote_name, arguments)

        return invoke

    async def _invoke(
        self,
        server_name: str,
        generation: str,
        remote_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        async with self._lifecycle_lock:
            state = self._servers.get(server_name)
            if state is None or state.closing or state.generation != generation:
                raise KeyError(f"MCP server {server_name!r} is not connected")
            state.active_calls += 1

        try:
            return await state.client.call_tool(remote_name, arguments, raise_on_error=False)
        except RemoteToolError:
            raise
        except Exception as error:
            raise RemoteToolError(
                {
                    "content": [],
                    "structured_content": {
                        "error": f"MCP transport error: {type(error).__name__}: {error}"
                    },
                    "is_error": True,
                }
            ) from error
        finally:
            async with state.condition:
                state.active_calls -= 1
                state.condition.notify_all()

    def _preflight_tools(
        self, tools: Iterable[Tool], *, ignored_ids: set[str] | None = None
    ) -> None:
        ignored_ids = ignored_ids or set()
        existing = [tool for tool in self._catalog.get_all_tools() if tool.id not in ignored_ids]
        existing_ids = {tool.id for tool in existing}
        existing_names = {tool.name for tool in existing}
        batch_ids: set[str] = set()
        batch_names: set[str] = set()
        for tool in tools:
            if (
                tool.id in existing_ids
                or tool.name in existing_names
                or tool.id in batch_ids
                or tool.name in batch_names
            ):
                raise ValueError(f"tool collision for id {tool.id!r} or name {tool.name!r}")
            batch_ids.add(tool.id)
            batch_names.add(tool.name)

    def _register_tools(self, tools: Iterable[Tool]) -> None:
        batch = list(tools)
        strict_register = getattr(self._catalog, "register_many_strict", None)
        try:
            if strict_register is not None:
                strict_register(batch)
            else:
                for tool in batch:
                    self._catalog.register(tool)
        except BaseException:
            self._unregister_tools(tool.id for tool in batch)
            raise

    def _unregister_tools(self, tools: Iterable[str] | dict[str, Tool]) -> None:
        for tool_id in tools:
            try:
                self._catalog.unregister(tool_id)
            except KeyError:
                pass

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"MCP server name must be a non-empty string without '/', got {name!r}"
            )

    @staticmethod
    def _client_source(name: str, source: Any) -> Any:
        if (
            isinstance(source, dict)
            and "mcpServers" not in source
            and ("url" in source or "command" in source)
        ):
            return {"mcpServers": {name: source}}
        return source
