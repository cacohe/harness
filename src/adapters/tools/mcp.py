from __future__ import annotations

import asyncio
import inspect
import json
import secrets
import threading
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from src.adapters.tools.catalog import ToolCatalog, ToolCatalogEntry
from src.adapters.tools.execution_store import ToolExecutionStore
from src.adapters.tools.local import LocalToolGateway
from src.core.artifacts import ArtifactExternalizer
from src.core.tools import ToolSpec

_MAX_SYNC_CALLS = 16
_sync_calls_lock = threading.Lock()
_sync_calls_active = 0
_late_cleanup_tasks: set[asyncio.Task[None]] = set()


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
        self._sync_timeout = self._timeout
        self._workspace = workspace_id
        self._servers: dict[str, _Server] = {}
        self._lifecycle = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._pending_adds = 0
        self._adds_idle = asyncio.Event()
        self._adds_idle.set()
        self._retired_closes: set[asyncio.Task[None]] = set()
        self._retired_failure: BaseException | None = None

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
                timeout=self._sync_timeout,
                late_result_cleanup=partial(_close_once, timeout=self._sync_timeout),
            )
            descriptors = await _await_method(client, "list_tools", timeout=self._sync_timeout)
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
                await _owned_close(client, self._sync_timeout)
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
            descriptors = await _await_method(
                state.client, "list_tools", timeout=self._sync_timeout
            )
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
        close_task: asyncio.Task[None] | None = None
        try:
            await self._wait_idle(state)
            async with self._lifecycle:
                if self._servers.get(name) is not state:
                    raise RuntimeError("MCP server changed during removal")
                replacement = dict(self._servers)
                del replacement[name]
                self._publish(replacement)
                self._servers = replacement
                close_task = self._start_close(state)
                self._retired_closes.add(close_task)
                close_task.add_done_callback(self._retired_close_done)
                committed = True
            await asyncio.shield(close_task)
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
                retired = tuple(self._retired_closes)
                task = asyncio.create_task(self._close_all(states, retired, self._retired_failure))
                self._close_task = task
        await asyncio.shield(task)

    async def _close_all(
        self,
        states: tuple[_Server, ...],
        retired: tuple[asyncio.Task[None], ...],
        retired_failure: BaseException | None,
    ) -> None:
        await self._adds_idle.wait()
        failure = retired_failure
        for task in retired:
            try:
                await asyncio.shield(task)
            except BaseException as error:
                if failure is None:
                    failure = error
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
            return await _await_method(
                state.client,
                "call_tool",
                tool_name,
                arguments,
                timeout=self._sync_timeout,
            )
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
        await asyncio.shield(self._start_close(state))

    def _start_close(self, state: _Server) -> asyncio.Task[None]:
        task = state.close_task
        if task is None:
            state.closed = True
            task = asyncio.create_task(_close_once(state.client, self._sync_timeout))
            state.close_task = task
        return task

    def _retired_close_done(self, task: asyncio.Task[None]) -> None:
        self._retired_closes.discard(task)
        try:
            failure = task.exception()
        except BaseException as error:
            failure = error
        if failure is not None and self._retired_failure is None:
            self._retired_failure = failure

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


async def _await_method(target: object, name: str, *args: object, timeout: float) -> object:
    method = getattr(target, name, None)
    if not callable(method):
        raise ValueError("MCP client contract is invalid")
    return await _invoke(method, *args, timeout=timeout)


async def _invoke(
    method: Callable[..., object],
    *args: object,
    timeout: float,
    late_result_cleanup: Callable[[object], object] | None = None,
) -> object:
    if inspect.iscoroutinefunction(method):
        return await method(*args)
    call = _SyncCall(method, args, late_result_cleanup, timeout)
    result = await call.wait(timeout)
    if inspect.isawaitable(result):
        result = await asyncio.wait_for(result, timeout)
    return result


class _SyncCall:
    def __init__(
        self,
        method: Callable[..., object],
        args: tuple[object, ...],
        late_result_cleanup: Callable[[object], object] | None,
        cleanup_timeout: float,
    ) -> None:
        self._method = method
        self._args = args
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[object] = self._loop.create_future()
        self._late_result_cleanup = late_result_cleanup
        self._cleanup_timeout = cleanup_timeout
        self._abandoned = False
        self._abandoned_signal = threading.Event()
        self._cleanup_scheduled = False
        self._start()

    def _start(self) -> None:
        global _sync_calls_active
        with _sync_calls_lock:
            if _sync_calls_active >= _MAX_SYNC_CALLS:
                raise TimeoutError("MCP synchronous call capacity is exhausted")
            _sync_calls_active += 1
        try:
            threading.Thread(
                target=self._run,
                name="harness-mcp-sync",
                daemon=True,
            ).start()
        except BaseException:
            with _sync_calls_lock:
                _sync_calls_active -= 1
            raise

    def _run(self) -> None:
        global _sync_calls_active
        result: object | None = None
        failure: BaseException | None = None
        cleaned = False
        try:
            result = self._method(*self._args)
        except BaseException as error:
            failure = error
        try:
            if (
                failure is None
                and result is not None
                and self._abandoned_signal.is_set()
                and self._late_result_cleanup is not None
            ):
                _fallback_late_close(result, self._cleanup_timeout)
                cleaned = True
            try:
                self._loop.call_soon_threadsafe(self._complete, result, failure, cleaned)
            except RuntimeError:
                if (
                    failure is None
                    and result is not None
                    and not cleaned
                    and self._late_result_cleanup is not None
                ):
                    _fallback_late_close(result, self._cleanup_timeout)
        finally:
            with _sync_calls_lock:
                _sync_calls_active -= 1

    def _complete(
        self,
        result: object | None,
        failure: BaseException | None,
        cleaned: bool,
    ) -> None:
        if self._abandoned:
            if failure is None and not cleaned:
                self._schedule_cleanup(result)
            return
        if failure is not None:
            self._future.set_exception(failure)
        else:
            self._future.set_result(result)

    async def wait(self, timeout: float) -> object:
        try:
            return await asyncio.wait_for(asyncio.shield(self._future), timeout)
        except (asyncio.CancelledError, TimeoutError):
            self._abandon()
            raise

    def _abandon(self) -> None:
        if self._abandoned:
            return
        self._abandoned = True
        self._abandoned_signal.set()
        if not self._future.done():
            self._future.cancel()
            return
        try:
            result = self._future.result()
        except BaseException:
            return
        self._schedule_cleanup(result)

    def _schedule_cleanup(self, result: object | None) -> None:
        if self._cleanup_scheduled:
            return
        self._cleanup_scheduled = True
        if result is None:
            return
        if not inspect.isawaitable(result) and self._late_result_cleanup is None:
            return
        task = asyncio.create_task(
            _finish_late_result(result, self._late_result_cleanup, self._cleanup_timeout)
        )
        _late_cleanup_tasks.add(task)
        task.add_done_callback(_late_cleanup_done)


async def _finish_late_result(
    result: object,
    cleanup: Callable[[object], object] | None,
    timeout: float,
) -> None:
    try:
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout)
        if cleanup is not None:
            outcome = cleanup(result)
            if inspect.isawaitable(outcome):
                await asyncio.wait_for(outcome, timeout)
    except BaseException:
        pass


def _late_cleanup_done(task: asyncio.Task[None]) -> None:
    _late_cleanup_tasks.discard(task)
    try:
        task.result()
    except BaseException:
        pass


def _fallback_late_close(result: object, timeout: float) -> None:
    try:
        if inspect.isawaitable(result):
            result = asyncio.run(asyncio.wait_for(result, timeout))
        closer = getattr(result, "aclose", None) or getattr(result, "close", None)
        if not callable(closer):
            return
        outcome = closer()
        if inspect.isawaitable(outcome):
            asyncio.run(asyncio.wait_for(outcome, timeout))
    except BaseException:
        pass


async def _close_once(client: object, timeout: float) -> None:
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if callable(closer):
        await _invoke(closer, timeout=timeout)


async def _owned_close(client: object, timeout: float) -> None:
    task = asyncio.create_task(_close_once(client, timeout))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
    await task
    if cancellation is not None:
        raise cancellation
