import asyncio
import threading
import time

import pytest
import src.adapters.tools.mcp as mcp_module
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.tools import InMemoryToolExecutionStore
from src.adapters.tools.mcp import MCPToolGateway
from src.core.commands import ToolCallRequest
from src.core.ids import RunId, SessionId
from src.core.tools import ToolApproval, ToolRequest, ToolResultStatus


class Client:
    def __init__(self):
        self.closed = 0

    async def list_tools(self):
        return [
            {
                "name": "read",
                "description": "read",
                "inputSchema": {"type": "object"},
                "annotations": {"readOnlyHint": True},
            }
        ]

    async def call_tool(self, name, arguments):
        return {"name": name, "arguments": arguments}

    async def aclose(self):
        self.closed += 1


def test_add_remove_is_atomic_and_closes_once() -> None:
    client = Client()

    async def scenario():
        gateway = MCPToolGateway(
            InMemoryToolExecutionStore(),
            PolicyArtifactExternalizer(
                InMemoryArtifactRepository(1024),
                ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
            ),
            client_factory=lambda _source: client,
        )
        await gateway.add_server("server", object())
        assert gateway._dynamic_catalog.specs[0].name == "server.read"
        await gateway.remove_server("server")
        assert gateway._dynamic_catalog.specs == ()

    asyncio.run(scenario())
    assert client.closed == 1


def _gateway(client, *, timeout_seconds=60.0, factory=None):
    return MCPToolGateway(
        InMemoryToolExecutionStore(),
        PolicyArtifactExternalizer(
            InMemoryArtifactRepository(1024),
            ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
        ),
        client_factory=factory or (lambda _source: client),
        timeout_seconds=timeout_seconds,
    )


def _request(gateway: MCPToolGateway) -> ToolRequest:
    spec = gateway._dynamic_catalog.specs[0]
    call = ToolCallRequest(
        "call-1",
        spec.name,
        "{}",
        requires_approval=spec.requires_approval,
        approval_scope=spec.approval_scope,
        workspace_id=spec.workspace_id,
    )
    return ToolRequest.from_call(
        call,
        run_id=RunId("run-1"),
        session_id=SessionId("session-1"),
        step_id="step-1",
        approval=ToolApproval.NOT_REQUIRED,
    )


def test_add_failure_rolls_back_and_closes_transport() -> None:
    class Broken(Client):
        async def list_tools(self):
            raise RuntimeError("bad catalog")

    client = Broken()

    async def scenario():
        gateway = _gateway(client)
        try:
            await gateway.add_server("server", object())
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid catalog was published")
        assert gateway._dynamic_catalog.specs == ()

    asyncio.run(scenario())
    assert client.closed == 1


def test_refresh_waits_for_active_lease_and_stale_generation_is_rejected() -> None:
    class Blocking(Client):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call_tool(self, name, arguments):
            self.started.set()
            await self.release.wait()
            return await super().call_tool(name, arguments)

    async def scenario():
        client = Blocking()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        old = gateway._servers["server"].generation
        call = asyncio.create_task(gateway._call("server", old, "read", {}))
        await client.started.wait()
        refresh = asyncio.create_task(gateway.refresh_server("server"))
        await asyncio.sleep(0)
        assert not refresh.done()
        client.release.set()
        await call
        await refresh
        assert gateway._servers["server"].generation != old
        try:
            await gateway._call("server", old, "read", {})
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale generation was called")
        await gateway.aclose()

    asyncio.run(scenario())


def test_refresh_failure_keeps_previous_catalog() -> None:
    class Flaky(Client):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def list_tools(self):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("refresh failed")
            return await super().list_tools()

    async def scenario():
        client = Flaky()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        before = gateway._dynamic_catalog.specs
        try:
            await gateway.refresh_server("server")
        except RuntimeError:
            pass
        assert gateway._dynamic_catalog.specs == before
        assert gateway._servers["server"].closing is False
        await gateway.aclose()

    asyncio.run(scenario())


def test_close_cleanup_survives_caller_cancellation() -> None:
    class SlowClose(Client):
        def __init__(self):
            super().__init__()
            self.closing = asyncio.Event()
            self.release = asyncio.Event()

        async def aclose(self):
            self.closing.set()
            await self.release.wait()
            await super().aclose()

    async def scenario():
        client = SlowClose()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        task = asyncio.create_task(gateway.aclose())
        await client.closing.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        client.release.set()
        await gateway.aclose()
        assert client.closed == 1

    asyncio.run(scenario())


def test_cancelled_refresh_and_remove_roll_back_closing_state() -> None:
    class Blocking(Client):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call_tool(self, name, arguments):
            self.started.set()
            await self.release.wait()
            return await super().call_tool(name, arguments)

    async def cancel_operation(operation):
        client = Blocking()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        call = asyncio.create_task(gateway._call("server", generation, "read", {}))
        await client.started.wait()
        lifecycle = asyncio.create_task(operation(gateway))
        await asyncio.sleep(0)
        lifecycle.cancel()
        try:
            await lifecycle
        except asyncio.CancelledError:
            pass
        assert gateway._servers["server"].closing is False
        client.release.set()
        await call
        await gateway.aclose()

    asyncio.run(cancel_operation(lambda gateway: gateway.refresh_server("server")))
    asyncio.run(cancel_operation(lambda gateway: gateway.remove_server("server")))


def test_synchronous_client_methods_run_off_the_event_loop() -> None:
    release = threading.Event()

    class SyncClient:
        def __init__(self):
            self.closed = 0

        def list_tools(self):
            release.wait(0.2)
            return [
                {
                    "name": "read",
                    "description": "read",
                    "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ]

        def call_tool(self, name, arguments):
            release.clear()
            threading.Timer(0.1, release.set).start()
            release.wait(0.2)
            return {"name": name, "arguments": arguments}

        def close(self):
            self.closed += 1

    async def count_ticks(task):
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        return ticks

    async def scenario():
        client = SyncClient()
        gateway = _gateway(client)
        threading.Timer(0.1, release.set).start()
        adding = asyncio.create_task(gateway.add_server("server", object()))
        add_ticks = await count_ticks(adding)
        await adding
        generation = gateway._servers["server"].generation
        calling = asyncio.create_task(gateway._call("server", generation, "read", {}))
        call_ticks = await count_ticks(calling)
        await calling
        await gateway.aclose()
        assert add_ticks >= 5
        assert call_ticks >= 5
        assert client.closed == 1

    asyncio.run(scenario())


def test_close_waits_for_refresh_remote_call_before_closing_client() -> None:
    class Refreshing(Client):
        def __init__(self):
            super().__init__()
            self.list_count = 0
            self.refresh_started = asyncio.Event()
            self.release_refresh = asyncio.Event()

        async def list_tools(self):
            self.list_count += 1
            if self.list_count > 1:
                self.refresh_started.set()
                await self.release_refresh.wait()
            return await super().list_tools()

    async def scenario():
        client = Refreshing()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        refresh = asyncio.create_task(gateway.refresh_server("server"))
        await client.refresh_started.wait()
        closing = asyncio.create_task(gateway.aclose())
        await asyncio.sleep(0.01)
        assert client.closed == 0
        client.release_refresh.set()
        outcomes = await asyncio.gather(refresh, closing, return_exceptions=True)
        assert isinstance(outcomes[0], RuntimeError)
        assert outcomes[1] is None
        assert client.closed == 1

    asyncio.run(scenario())


def test_close_waits_for_inflight_add_to_close_its_unpublished_client() -> None:
    class Adding(Client):
        def __init__(self):
            super().__init__()
            self.list_started = asyncio.Event()
            self.release_list = asyncio.Event()

        async def list_tools(self):
            self.list_started.set()
            await self.release_list.wait()
            return await super().list_tools()

    async def scenario():
        client = Adding()
        gateway = _gateway(client)
        adding = asyncio.create_task(gateway.add_server("server", object()))
        await client.list_started.wait()
        closing = asyncio.create_task(gateway.aclose())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.05)
        assert client.closed == 0
        client.release_list.set()
        outcomes = await asyncio.gather(adding, closing, return_exceptions=True)
        assert isinstance(outcomes[0], ValueError)
        assert outcomes[1] is None
        assert client.closed == 1

    asyncio.run(scenario())


def test_cancelled_add_tracks_late_synchronous_factory_client_cleanup() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()

    class Factory:
        def __init__(self, client):
            self.client = client

        def __call__(self, _source):
            factory_started.set()
            release_factory.wait(1)
            return self.client

    async def scenario():
        client = Client()
        gateway = _gateway(client)
        gateway._factory = Factory(client)
        adding = asyncio.create_task(gateway.add_server("server", object()))
        await asyncio.to_thread(factory_started.wait, 1)
        adding.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(adding, 0.2)
            assert client.closed == 0
        finally:
            release_factory.set()
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.01)
        assert client.closed == 1

    asyncio.run(scenario())


def test_cancelled_add_closes_owned_client_without_waiting_for_sync_discovery() -> None:
    discovery_started = threading.Event()
    release_discovery = threading.Event()

    class Discovering(Client):
        def list_tools(self):
            discovery_started.set()
            release_discovery.wait(1)
            return [
                {
                    "name": "read",
                    "description": "read",
                    "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ]

    async def scenario():
        client = Discovering()
        gateway = _gateway(client)
        adding = asyncio.create_task(gateway.add_server("server", object()))
        await asyncio.to_thread(discovery_started.wait, 1)
        adding.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(adding, 0.2)
            assert client.closed == 1
        finally:
            release_discovery.set()

    asyncio.run(scenario())


def test_close_waits_for_concurrent_removed_server_close() -> None:
    class BlockingClose(Client):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self):
            self.close_started.set()
            await self.release_close.wait()
            self.closed += 1

    async def scenario():
        client = BlockingClose()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        removing = asyncio.create_task(gateway.remove_server("server"))
        await client.close_started.wait()
        closing = asyncio.create_task(gateway.aclose())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.05)
        client.release_close.set()
        assert await removing is None
        assert await closing is None
        assert await gateway.aclose() is None
        assert client.closed == 1

    asyncio.run(scenario())


def test_close_observes_concurrent_removed_server_close_failure() -> None:
    class FailingClose(Client):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self):
            self.close_started.set()
            await self.release_close.wait()
            raise RuntimeError("close failed")

    async def scenario():
        client = FailingClose()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        removing = asyncio.create_task(gateway.remove_server("server"))
        await client.close_started.wait()
        closing = asyncio.create_task(gateway.aclose())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.05)
        client.release_close.set()
        outcomes = await asyncio.gather(removing, closing, return_exceptions=True)
        assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
        with pytest.raises(RuntimeError, match="close failed"):
            await gateway.aclose()

    asyncio.run(scenario())


def test_close_waits_for_removed_server_close_after_remove_is_cancelled() -> None:
    class BlockingClose(Client):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self):
            self.close_started.set()
            await self.release_close.wait()
            self.closed += 1

    async def scenario():
        client = BlockingClose()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        removing = asyncio.create_task(gateway.remove_server("server"))
        await client.close_started.wait()
        removing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await removing
        closing = asyncio.create_task(gateway.aclose())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.05)
        client.release_close.set()
        assert await closing is None
        assert client.closed == 1

    asyncio.run(scenario())


def test_blocked_synchronous_call_times_out_and_shutdown_is_bounded() -> None:
    call_started = threading.Event()
    release_call = threading.Event()

    class BlockingCall:
        def __init__(self):
            self.closed = 0

        def list_tools(self):
            return [
                {
                    "name": "read",
                    "description": "read",
                    "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ]

        def call_tool(self, _name, _arguments):
            call_started.set()
            release_call.wait()
            return {"late": True}

        def close(self):
            self.closed += 1

    async def scenario():
        client = BlockingCall()
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        started = time.monotonic()
        executing = asyncio.create_task(gateway.execute(_request(gateway)))
        assert await asyncio.to_thread(call_started.wait, 1)
        closing = asyncio.create_task(gateway.aclose())
        result = await asyncio.wait_for(executing, 0.3)
        await asyncio.wait_for(closing, 0.3)
        assert result.status is ToolResultStatus.TIMED_OUT
        assert time.monotonic() - started < 0.3
        assert client.closed == 1

    watchdog = threading.Timer(0.6, release_call.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release_call.set()
        watchdog.cancel()


def test_blocked_synchronous_factory_times_out_and_late_client_is_closed() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    client = Client()

    def factory(_source):
        factory_started.set()
        release_factory.wait()
        return client

    async def scenario():
        gateway = _gateway(client, timeout_seconds=0.05, factory=factory)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.add_server("server", object()), 0.3)
        assert time.monotonic() - started < 0.3
        release_factory.set()
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.01)
        assert client.closed == 1
        assert await asyncio.wait_for(gateway.aclose(), 0.3) is None

    watchdog = threading.Timer(0.6, release_factory.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release_factory.set()
        watchdog.cancel()


def test_blocked_synchronous_discovery_times_out_and_owned_client_is_closed() -> None:
    discovery_started = threading.Event()
    release_discovery = threading.Event()

    class BlockingDiscovery:
        def __init__(self):
            self.closed = 0

        def list_tools(self):
            discovery_started.set()
            release_discovery.wait()
            return []

        def close(self):
            self.closed += 1

    async def scenario():
        client = BlockingDiscovery()
        gateway = _gateway(client, timeout_seconds=0.05)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.add_server("server", object()), 0.3)
        assert time.monotonic() - started < 0.3
        assert client.closed == 1
        assert await asyncio.wait_for(gateway.aclose(), 0.3) is None

    watchdog = threading.Timer(0.6, release_discovery.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release_discovery.set()
        watchdog.cancel()


def test_blocked_synchronous_close_fails_bounded_once_and_is_idempotent() -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingClose:
        def __init__(self):
            self.close_calls = 0

        def list_tools(self):
            return []

        def close(self):
            self.close_calls += 1
            close_started.set()
            release_close.wait()

    async def scenario():
        client = BlockingClose()
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.aclose(), 0.3)
        assert time.monotonic() - started < 0.3
        assert close_started.is_set()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.aclose(), 0.1)
        assert client.close_calls == 1

    watchdog = threading.Timer(0.6, release_close.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release_close.set()
        watchdog.cancel()


def test_cancelled_synchronous_call_releases_lease_and_shutdown_completes() -> None:
    call_started = threading.Event()
    release_call = threading.Event()

    class BlockingCall:
        def __init__(self):
            self.closed = 0

        def list_tools(self):
            return [
                {
                    "name": "read",
                    "description": "read",
                    "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ]

        def call_tool(self, _name, _arguments):
            call_started.set()
            release_call.wait()
            return {}

        def close(self):
            self.closed += 1

    async def scenario():
        client = BlockingCall()
        gateway = _gateway(client, timeout_seconds=0.2)
        await gateway.add_server("server", object())
        executing = asyncio.create_task(gateway.execute(_request(gateway)))
        assert await asyncio.to_thread(call_started.wait, 1)
        executing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(executing, 0.2)
        assert await asyncio.wait_for(gateway.aclose(), 0.2) is None
        assert client.closed == 1

    watchdog = threading.Timer(0.6, release_call.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release_call.set()
        watchdog.cancel()


def test_synchronous_worker_count_is_bounded_and_late_clients_are_owned() -> None:
    release_factory = threading.Event()
    created: list[Client] = []
    created_lock = threading.Lock()

    def factory(_source):
        client = Client()
        with created_lock:
            created.append(client)
        release_factory.wait()
        return client

    async def scenario():
        gateway = _gateway(Client(), timeout_seconds=0.05, factory=factory)
        outcomes = await asyncio.gather(
            *(gateway.add_server(f"server-{index}", object()) for index in range(32)),
            return_exceptions=True,
        )
        assert all(isinstance(outcome, TimeoutError) for outcome in outcomes)
        workers = [thread for thread in threading.enumerate() if thread.name == "harness-mcp-sync"]
        assert len(workers) <= mcp_module._MAX_SYNC_CALLS
        assert len(created) <= mcp_module._MAX_SYNC_CALLS
        release_factory.set()
        for _ in range(200):
            if created and all(client.closed == 1 for client in created):
                break
            await asyncio.sleep(0.01)
        assert created and all(client.closed == 1 for client in created)
        assert await asyncio.wait_for(gateway.aclose(), 0.2) is None

    watchdog = threading.Timer(0.6, release_factory.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release_factory.set()
        watchdog.cancel()
