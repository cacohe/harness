from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import threading
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from src.adapters.tools.catalog import ToolCatalog, ToolCatalogEntry
from src.adapters.tools.execution_store import ToolExecutionStore
from src.adapters.tools.local import LocalToolGateway
from src.core.artifacts import ArtifactExternalizer
from src.core.tools import ToolSpec

_MAX_SYNC_CALLS_PER_GATEWAY = 4
_MAX_SYNC_CLOSES_PER_GATEWAY = 4
_PROVIDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_UNSAFE_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_late_cleanup_tasks: set[asyncio.Task[None]] = set()


class _SyncLimiter:
    def __init__(self, limit: int = _MAX_SYNC_CALLS_PER_GATEWAY) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._active = 0

    def acquire(self) -> None:
        with self._lock:
            if self._active >= self._limit:
                raise TimeoutError("MCP synchronous call capacity is exhausted")
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active -= 1


@dataclass(slots=True)
class _Server:
    client: object
    generation: str
    entries: tuple[ToolCatalogEntry, ...]
    cleanup: _LiveClientCleanup
    active: int = 0
    closing: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    closed: bool = False
    close_task: asyncio.Task[None] | None = None
    refresh_token: object | None = None


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
        self._sync_limiter = _SyncLimiter()
        # Cleanup must remain available when all operation workers are stuck.
        # Neither limiter queues work: at most eight daemon workers per gateway,
        # even if operations and cleanup both block permanently.
        self._cleanup_limiter = _SyncLimiter(_MAX_SYNC_CLOSES_PER_GATEWAY)
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
        discovery: _Invocation | None = None
        try:
            client = await _invoke(
                self._factory,
                source,
                timeout=self._sync_timeout,
                limiter=self._sync_limiter,
                late_result_cleanup=partial(
                    _close_once,
                    timeout=self._sync_timeout,
                    limiter=self._cleanup_limiter,
                ),
            )
            client = await _enter_async_context(
                client, timeout=self._sync_timeout, limiter=self._sync_limiter
            )
            discovery = _Invocation(
                _client_method(client, "list_tools"),
                (),
                self._sync_limiter,
                None,
                self._sync_timeout,
                abandoned_client=client,
            )
            descriptors = await discovery.wait(self._sync_timeout)
            normalized = _normalize_descriptors(name, descriptors)
            generation = _catalog_generation(name, normalized)
            entries = self._entries(name, generation, client, normalized)
            async with self._lifecycle:
                if self._closed or name in self._servers:
                    raise ValueError("MCP server cannot be added")
                cleanup = _LiveClientCleanup(client, self._sync_timeout, self._cleanup_limiter)
                state = _Server(client, generation, entries, cleanup)
                replacement = {**self._servers, name: state}
                self._publish(replacement)
                self._servers = replacement
                if isinstance(client, _AsyncContextClient):
                    client.on_terminated(partial(self._async_client_terminated, name, state))
        except BaseException:
            if client is not None:
                if discovery is not None and discovery.abandoned:
                    self._defer_close_until_settled(client, discovery)
                else:
                    await _owned_close(client, self._sync_timeout, self._cleanup_limiter)
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
            token = state.refresh_token = object()
            async with state.condition:
                state.active += 1
        committed = False
        invocation: _Invocation | None = None
        lease_transferred = False
        try:
            await asyncio.wait_for(self._wait_active(state, 1), self._sync_timeout)
            invocation = _Invocation(
                _client_method(state.client, "list_tools"),
                (),
                self._sync_limiter,
                None,
                self._sync_timeout,
                client_cleanup=state.cleanup,
            )
            invocation.settled.add_done_callback(
                partial(self._refresh_done, name, state, token, invocation)
            )
            lease_transferred = True
            descriptors = await invocation.wait(self._sync_timeout)
            normalized = _normalize_descriptors(name, descriptors)
            generation = _catalog_generation(name, normalized)
            entries = self._entries(name, generation, state.client, normalized)
            async with self._lifecycle:
                current = self._servers.get(name)
                if current is not state or self._closed:
                    raise RuntimeError("MCP server changed during refresh")
                replacement = dict(self._servers)
                replacement[name] = _Server(state.client, generation, entries, state.cleanup)
                self._publish(replacement)
                self._servers = replacement
                committed = True
        finally:
            if not lease_transferred:
                await self._release_active(state)
            if not committed and (
                invocation is None or not invocation.abandoned or invocation.settled.done()
            ):
                await self._restore_refresh(name, state, token)

    def _refresh_done(
        self,
        name: str,
        state: _Server,
        token: object,
        invocation: _Invocation,
        _future: asyncio.Future[None],
    ) -> None:
        async def release() -> None:
            await self._release_active(state)
            if invocation.abandoned:
                await self._restore_refresh(name, state, token)

        asyncio.create_task(release())

    async def _restore_refresh(self, name: str, state: _Server, token: object) -> None:
        async with self._lifecycle:
            if (
                not self._closed
                and self._servers.get(name) is state
                and state.refresh_token is token
            ):
                state.refresh_token = None
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
            await asyncio.wait_for(self._wait_idle(state), self._sync_timeout)
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
                    state.cleanup.request_close()
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
        async with self._lifecycle:
            retired = tuple({*retired, *self._retired_closes})
            if retired_failure is None:
                retired_failure = self._retired_failure
        failure = retired_failure
        for task in retired:
            try:
                await asyncio.wait_for(asyncio.shield(task), self._sync_timeout)
            except BaseException as error:
                if failure is None:
                    failure = error
        for state in states:
            try:
                await asyncio.wait_for(self._wait_idle(state), self._sync_timeout)
                await self._shielded_close(state)
            except BaseException as error:
                if failure is None:
                    failure = error
                if state.active > 0:
                    self._defer_close_until_idle(state)
        if failure is not None:
            raise failure

    def _entries(
        self,
        name: str,
        generation: str,
        client: object,
        descriptors: tuple[tuple[str, str, str, str], ...],
    ) -> tuple[ToolCatalogEntry, ...]:
        entries: list[ToolCatalogEntry] = []
        for remote_name, public_segment, description, schema_json in descriptors:
            public_name = _public_tool_name(name, public_segment)
            mutating = True
            workspace = f"{self._workspace}-{generation}"
            spec = ToolSpec(
                public_name,
                description,
                schema_json,
                requires_approval=True,
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
            method = _client_method(state.client, "call_tool")
            invocation = _Invocation(
                method,
                (tool_name, arguments),
                self._sync_limiter,
                None,
                self._sync_timeout,
                client_cleanup=state.cleanup,
            )
        except BaseException:
            await self._release_active(state)
            raise
        invocation.settled.add_done_callback(
            lambda _future: asyncio.create_task(self._release_active(state))
        )
        result = await invocation.wait(self._sync_timeout)
        if _value(result, "isError", _value(result, "is_error", False)) is True:
            raise RuntimeError("MCP tool returned an error result")
        return result

    @staticmethod
    async def _release_active(state: _Server) -> None:
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
            task = asyncio.create_task(state.cleanup.close())
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

    def _track_retired_close(self, task: asyncio.Task[None]) -> None:
        self._retired_closes.add(task)
        task.add_done_callback(self._retired_close_done)

    def _defer_close_until_settled(self, client: object, invocation: _Invocation) -> None:
        async def close_later() -> None:
            await asyncio.shield(invocation.settled)
            if invocation.worker_cleanup:
                failure = await asyncio.shield(invocation.cleanup_complete)
                if failure is not None:
                    raise failure
            else:
                await _close_once(client, self._sync_timeout, self._cleanup_limiter)

        self._track_retired_close(asyncio.create_task(close_later()))

    def _defer_close_until_idle(self, state: _Server) -> None:
        async def close_later() -> None:
            await self._wait_idle(state)
            await self._shielded_close(state)

        if state.close_task is None:
            self._track_retired_close(asyncio.create_task(close_later()))

    def _async_client_terminated(self, name: str, state: _Server) -> None:
        async def retire() -> None:
            async with self._lifecycle:
                if self._servers.get(name) is not state:
                    return
                state.closing = True
                replacement = dict(self._servers)
                del replacement[name]
                self._publish(replacement)
                self._servers = replacement
            self._defer_close_until_idle(state)

        asyncio.create_task(retire())

    def _publish(self, servers: dict[str, _Server]) -> None:
        entries = tuple(entry for state in servers.values() for entry in state.entries)
        self._dynamic_catalog.replace(entries)


@dataclass(slots=True)
class _AsyncContextCall:
    name: str
    args: tuple[object, ...]
    result: asyncio.Future[object]
    started: bool = False


class _AsyncContextClient:
    def __init__(self, manager: object, timeout: float, limiter: _SyncLimiter) -> None:
        self._manager = manager
        self._loop = asyncio.get_running_loop()
        self._ready: asyncio.Future[None] = self._loop.create_future()
        self._queue: asyncio.Queue[_AsyncContextCall | None] = asyncio.Queue()
        self._closed = False
        self._terminal_error: BaseException | None = None
        self._timeout = timeout
        self._limiter = limiter
        self._termination_callbacks: list[Callable[[], object]] = []
        self._owner = asyncio.create_task(self._run())
        self._owner.add_done_callback(self._owner_done)

    def on_terminated(self, callback: Callable[[], object]) -> None:
        if self._owner.done():
            callback()
        else:
            self._termination_callbacks.append(callback)

    def _owner_done(self, _task: asyncio.Task[None]) -> None:
        callbacks, self._termination_callbacks = self._termination_callbacks, []
        for callback in callbacks:
            callback()

    def __getattr__(self, name: str) -> object:
        async def invoke(*args: object) -> object:
            if self._closed or self._owner.done():
                raise RuntimeError("MCP async context client is closed")
            result: asyncio.Future[object] = self._loop.create_future()
            call = _AsyncContextCall(name, args, result)
            await self._queue.put(call)
            try:
                return await result
            except asyncio.CancelledError:
                if not result.done():
                    result.cancel()
                if call.started:
                    self._owner.cancel()
                raise

        return invoke

    async def start(self, timeout: float) -> _AsyncContextClient:
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), timeout)
        except BaseException:
            self._closed = True
            self._queue.put_nowait(None)
            self._owner.cancel()
            if not self._owner.done():
                _late_cleanup_tasks.add(self._owner)
                self._owner.add_done_callback(_late_cleanup_done)
            raise
        return self

    async def aclose(self) -> None:
        if self._closed:
            await asyncio.shield(self._owner)
        else:
            self._closed = True
            await self._queue.put(None)
            try:
                await asyncio.shield(self._owner)
            except asyncio.CancelledError:
                self._owner.cancel()
                raise
        if self._terminal_error is not None:
            raise self._terminal_error

    async def _run(self) -> None:
        entered = False
        client: object | None = None
        current: asyncio.Future[object] | None = None
        failure: BaseException | None = None
        try:
            enter = getattr(self._manager, "__aenter__")
            outcome = enter()
            client = await outcome if inspect.isawaitable(outcome) else outcome
            entered = True
            self._ready.set_result(None)
            while True:
                call = await self._queue.get()
                if call is None:
                    break
                current = call.result
                if current.cancelled():
                    current = None
                    continue
                call.started = True
                try:
                    method = _client_method(client, call.name)
                    if inspect.iscoroutinefunction(method):
                        value = await method(*call.args)
                    else:
                        value = await _Invocation(
                            method,
                            call.args,
                            self._limiter,
                            None,
                            self._timeout,
                        ).wait(self._timeout)
                except asyncio.CancelledError as error:
                    if not current.done():
                        current.cancel()
                    failure = error
                    raise
                except BaseException as error:
                    if not current.done():
                        current.set_exception(error)
                else:
                    if not current.done():
                        current.set_result(value)
                finally:
                    current = None
        except asyncio.CancelledError as error:
            failure = error
        except BaseException as error:
            failure = error
            self._terminal_error = error
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            if current is not None and not current.done():
                current.set_exception(RuntimeError("MCP async context client stopped"))
            if entered:
                try:
                    exit_context = getattr(self._manager, "__aexit__")
                    outcome = exit_context(
                        type(failure) if failure is not None else None,
                        failure,
                        failure.__traceback__ if failure is not None else None,
                    )
                    if inspect.isawaitable(outcome):
                        await outcome
                except BaseException as error:
                    self._terminal_error = error
            if not self._ready.done():
                self._ready.set_exception(
                    self._terminal_error or RuntimeError("MCP async context client stopped")
                )
            while not self._queue.empty():
                pending = self._queue.get_nowait()
                if pending is not None and not pending.result.done():
                    pending.result.set_exception(RuntimeError("MCP async context client stopped"))


async def _enter_async_context(client: object, *, timeout: float, limiter: _SyncLimiter) -> object:
    enter = getattr(client, "__aenter__", None)
    exit_context = getattr(client, "__aexit__", None)
    if enter is None and exit_context is None:
        return client
    if not callable(enter) or not callable(exit_context):
        raise ValueError("MCP async context manager contract is invalid")
    return await _AsyncContextClient(client, timeout, limiter).start(timeout)


def _server_name(name: object) -> None:
    if type(name) is not str or _PROVIDER_NAME.fullmatch(name) is None:
        raise ValueError("MCP server name is invalid")


def _public_tool_name(server_name: str, public_segment: str) -> str:
    if _PROVIDER_NAME.fullmatch(public_segment) is None:
        raise ValueError("MCP tool name is invalid")
    public_name = f"{server_name}__{public_segment}"
    if len(public_name.encode("ascii")) > 64:
        raise ValueError("MCP public tool name is invalid")
    return public_name


def _normalize_descriptors(
    server_name: str, descriptors: object
) -> tuple[tuple[str, str, str, str], ...]:
    if type(descriptors) is not list or len(descriptors) > 128:
        raise ValueError("MCP tool catalog is invalid")
    normalized: list[tuple[str, str, str, str]] = []
    for descriptor in descriptors:
        remote_name = _value(descriptor, "name")
        description = _value(descriptor, "description", "")
        schema = _value(descriptor, "inputSchema", _value(descriptor, "input_schema", {}))
        if (
            type(remote_name) is not str
            or not remote_name
            or len(remote_name.encode("utf-8")) > 1_024
            or type(description) is not str
            or type(schema) is not dict
        ):
            raise ValueError("MCP tool descriptor is invalid")
        public_segment = _normalized_tool_segment(server_name, remote_name)
        schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        normalized.append((remote_name, public_segment, description, schema_json))
    if len({item[0] for item in normalized}) != len(normalized) or len(
        {item[1] for item in normalized}
    ) != len(normalized):
        raise ValueError("MCP tool names are not unique")
    return tuple(normalized)


def _normalized_tool_segment(server_name: str, remote_name: str) -> str:
    available = 64 - len(server_name.encode("ascii")) - 2
    if available < 1:
        raise ValueError("MCP server name leaves no room for tool names")
    if _PROVIDER_NAME.fullmatch(remote_name) is not None and len(remote_name) <= available:
        return remote_name
    segment = _UNSAFE_TOOL_NAME.sub("_", remote_name)
    if not segment:
        segment = "tool"
    if not segment[0].isalnum() or not segment[0].isascii():
        segment = f"tool_{segment}"
    if _PROVIDER_NAME.fullmatch(segment) is None or len(segment.encode("ascii")) > available:
        digest = hashlib.sha256(remote_name.encode("utf-8")).hexdigest()[:8]
        if available <= 9:
            segment = f"t{digest}"[:available]
            _public_tool_name(server_name, segment)
            return segment
        base = _UNSAFE_TOOL_NAME.sub("_", segment) or "tool"
        if not base[0].isalnum() or not base[0].isascii():
            base = f"tool_{base}"
        segment = f"{base[: available - 9]}_{digest}"
    _public_tool_name(server_name, segment)
    return segment


def _catalog_generation(name: str, descriptors: tuple[tuple[str, str, str, str], ...]) -> str:
    payload = json.dumps(
        [name, descriptors], ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _value(value: object, name: str, default: object = ...) -> object:
    if type(value) is dict:
        if default is ...:
            return value[name]
        return value.get(name, default)
    if default is ...:
        return getattr(value, name)
    return getattr(value, name, default)


def _client_method(target: object, name: str) -> Callable[..., object]:
    method = getattr(target, name, None)
    if not callable(method):
        raise ValueError("MCP client contract is invalid")
    return method


async def _invoke(
    method: Callable[..., object],
    *args: object,
    timeout: float,
    limiter: _SyncLimiter,
    late_result_cleanup: Callable[[object], object] | None = None,
) -> object:
    return await _Invocation(
        method,
        args,
        limiter,
        late_result_cleanup,
        timeout,
    ).wait(timeout)


class _Invocation:
    def __init__(
        self,
        method: Callable[..., object],
        args: tuple[object, ...],
        limiter: _SyncLimiter,
        late_result_cleanup: Callable[[object], object] | None,
        cleanup_timeout: float,
        *,
        abandoned_client: object | None = None,
        client_cleanup: _LiveClientCleanup | None = None,
    ) -> None:
        self._method = method
        self._args = args
        self._limiter = limiter
        self._loop = asyncio.get_running_loop()
        self._future: asyncio.Future[object] = self._loop.create_future()
        self.settled: asyncio.Future[None] = self._loop.create_future()
        self._late_result_cleanup = late_result_cleanup
        self._cleanup_timeout = cleanup_timeout
        self._abandoned = False
        self._abandoned_signal = threading.Event()
        self._cleanup_scheduled = False
        self._abandoned_client = abandoned_client
        self._public_finished = threading.Event()
        self.worker_cleanup = abandoned_client is not None and not inspect.iscoroutinefunction(
            method
        )
        self.cleanup_complete: asyncio.Future[BaseException | None] = self._loop.create_future()
        self._client_cleanup = client_cleanup
        self._usage_released = False
        self._async_task: asyncio.Future[object] | None = None
        if client_cleanup is not None:
            client_cleanup.acquire()
        try:
            if inspect.iscoroutinefunction(method):
                task = asyncio.create_task(method(*args))
                self._async_task = task
                task.add_done_callback(self._async_done)
            else:
                self._start_sync()
        except BaseException:
            if client_cleanup is not None:
                client_cleanup.release_on_loop()
            raise

    @property
    def abandoned(self) -> bool:
        return self._abandoned

    def _start_sync(self) -> None:
        self._limiter.acquire()
        try:
            threading.Thread(
                target=self._run_sync,
                name="harness-mcp-sync",
                daemon=True,
            ).start()
        except BaseException:
            self._limiter.release()
            raise

    def _run_sync(self) -> None:
        result: object | None = None
        failure: BaseException | None = None
        cleaned = False
        try:
            result = self._method(*self._args)
        except BaseException as error:
            failure = error
        try:
            if self._client_cleanup is not None and not inspect.isawaitable(result):
                self._usage_released = True
                self._client_cleanup.release_on_worker()
            if self.worker_cleanup and not inspect.isawaitable(result):
                # Retain the original bounded worker until the public waiter has
                # accepted the result or transferred abandoned-client ownership.
                try:
                    self._loop.call_soon_threadsafe(self._complete, result, failure, False)
                    while not self._public_finished.wait(self._cleanup_timeout):
                        if self._loop.is_closed():
                            break
                    accepted = self._public_finished.is_set()
                except RuntimeError:
                    accepted = False
                cleanup_failure = None
                if not accepted or self._abandoned_signal.is_set():
                    cleanup_failure = _WorkerCleanup(
                        self._abandoned_client, self._loop, self._cleanup_timeout
                    ).run()
                try:
                    self._loop.call_soon_threadsafe(
                        self.cleanup_complete.set_result, cleanup_failure
                    )
                except RuntimeError:
                    pass
                return
            self.worker_cleanup = False
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
            self._limiter.release()

    def _async_done(self, task: asyncio.Future[object]) -> None:
        try:
            result = task.result()
        except BaseException as error:
            self._complete(None, error, False)
        else:
            self._complete(result, None, False)

    def _complete(
        self,
        result: object | None,
        failure: BaseException | None,
        cleaned: bool,
    ) -> None:
        if failure is None and result is not None and inspect.isawaitable(result):
            task = asyncio.ensure_future(result)
            self._async_task = task
            task.add_done_callback(self._async_done)
            return
        if self._client_cleanup is not None and not self._usage_released:
            self._usage_released = True
            self._client_cleanup.release_on_loop()
        if not self.settled.done():
            self.settled.set_result(None)
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
        finally:
            self._public_finished.set()

    def _abandon(self) -> None:
        if self._abandoned:
            return
        self._abandoned = True
        self._abandoned_signal.set()
        if self._async_task is not None and not self._async_task.done():
            self._async_task.cancel()
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


class _WorkerCleanup:
    """Single close ownership retained by an existing bounded daemon worker.

    Async close uses the origin loop while it is available. If that loop stops
    before accepting the close, the same worker uses a private loop. Clients
    whose close requires an already-destroyed loop cannot be repaired by a
    gateway; such a close failure is retained, never retried concurrently.
    """

    def __init__(self, client: object, loop: asyncio.AbstractEventLoop, timeout: float) -> None:
        self._client = client
        self._loop = loop
        self._timeout = timeout
        self._lock = threading.Lock()
        self._claimed = False
        self._done = threading.Event()
        self._failure: BaseException | None = None

    def _claim(self) -> bool:
        with self._lock:
            if self._claimed:
                return False
            self._claimed = True
            return True

    def run(self) -> BaseException | None:
        closer = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if not callable(closer):
            return None
        try:
            if not inspect.iscoroutinefunction(closer):
                outcome = closer()
                if not inspect.isawaitable(outcome):
                    return None

                def pending_close() -> object:
                    return outcome

                closer = pending_close
            try:
                self._loop.call_soon_threadsafe(self._start, closer)
            except RuntimeError:
                pass
            else:
                if self._done.wait(self._timeout) and self._claimed:
                    return self._failure
            if self._claim():
                asyncio.run(self._await_close(closer))
                return None
            return self._failure if self._done.is_set() else TimeoutError("MCP cleanup timed out")
        except BaseException as error:
            return error

    async def _await_close(self, closer: Callable[..., object]) -> None:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await asyncio.wait_for(outcome, self._timeout)

    def _start(self, closer: Callable[..., object]) -> None:
        task = asyncio.create_task(self._on_loop_close(closer))
        _late_cleanup_tasks.add(task)
        task.add_done_callback(self._finished)

    async def _on_loop_close(self, closer: Callable[..., object]) -> None:
        if self._claim():
            await self._await_close(closer)

    def _finished(self, task: asyncio.Task[None]) -> None:
        _late_cleanup_tasks.discard(task)
        try:
            task.result()
        except BaseException as error:
            self._failure = error
        self._done.set()


class _LiveClientCleanup:
    """Underlying-use ownership independent of the public loop's lease tasks."""

    def __init__(self, client: object, timeout: float, limiter: _SyncLimiter) -> None:
        self._client = client
        self._timeout = timeout
        self._limiter = limiter
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._users = 0
        self._closing = False
        self._claimed = False
        self._completion: asyncio.Future[BaseException | None] = self._loop.create_future()
        self.failure: BaseException | None = None

    def acquire(self) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("MCP client is closing")
            self._users += 1

    def _release(self) -> bool:
        with self._lock:
            self._users -= 1
            return self._claim_idle()

    def _claim_idle(self) -> bool:
        if self._closing and self._users == 0 and not self._claimed:
            self._claimed = True
            return True
        return False

    def release_on_worker(self) -> None:
        if self._release():
            # Reuse the final invocation's worker, including its capacity slot.
            self.failure = _WorkerCleanup(self._client, self._loop, self._timeout).run()
            try:
                self._loop.call_soon_threadsafe(self._completion.set_result, self.failure)
            except RuntimeError:
                pass

    def release_on_loop(self) -> None:
        if self._release():
            self._start_on_loop()

    def request_close(self) -> None:
        with self._lock:
            self._closing = True
            start = self._claim_idle()
        if start:
            self._start_on_loop()

    def _start_on_loop(self) -> None:
        task = asyncio.create_task(self._close_on_loop())
        _late_cleanup_tasks.add(task)
        task.add_done_callback(_late_cleanup_done)

    async def _close_on_loop(self) -> None:
        try:
            await _close_once(self._client, self._timeout, self._limiter)
        except BaseException as error:
            self.failure = error
        self._completion.set_result(self.failure)

    async def close(self) -> None:
        self.request_close()
        failure = await asyncio.wait_for(asyncio.shield(self._completion), self._timeout)
        if failure is not None:
            raise failure


async def _close_once(client: object, timeout: float, limiter: _SyncLimiter) -> None:
    closer = getattr(client, "aclose", None) or getattr(client, "close", None)
    if callable(closer):
        await _invoke(closer, timeout=timeout, limiter=limiter)


async def _owned_close(client: object, timeout: float, limiter: _SyncLimiter) -> None:
    task = asyncio.create_task(_close_once(client, timeout, limiter))
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
    await task
    if cancellation is not None:
        raise cancellation
