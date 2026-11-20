import asyncio
import threading
import time
from dataclasses import dataclass

import anyio
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
        assert gateway._dynamic_catalog.specs[0].name == "server__read"
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


def _request(
    gateway: MCPToolGateway,
    *,
    spec_index: int = 0,
    call_id: str = "call-1",
    step_id: str = "step-1",
) -> ToolRequest:
    spec = gateway._dynamic_catalog.specs[spec_index]
    call = ToolCallRequest(
        call_id,
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
        step_id=step_id,
        approval=(ToolApproval.APPROVED if spec.requires_approval else ToolApproval.NOT_REQUIRED),
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
            self.catalog_calls = 0

        async def list_tools(self):
            self.catalog_calls += 1
            descriptors = await super().list_tools()
            if self.catalog_calls > 1:
                descriptors[0]["description"] = "changed"
            return descriptors

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


def test_refresh_keeps_transport_termination_bound_to_current_generation() -> None:
    class Manager:
        async def __aenter__(self):
            return Client()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def scenario():
        gateway = _gateway(Client(), factory=lambda _source: Manager())
        await gateway.add_server("server", object())
        await gateway.refresh_server("server")
        proxy = gateway._servers["server"].client
        proxy._owner.cancel()
        await proxy._owner
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert gateway._dynamic_catalog.specs == ()
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


def test_cancelled_add_defers_close_until_sync_discovery_settles() -> None:
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
            assert client.closed == 0
        finally:
            release_discovery.set()
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.01)
        assert client.closed == 1

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
            self.in_call = False
            self.closed_while_in_call = False

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
            self.in_call = True
            try:
                release_call.wait()
                return {"late": True}
            finally:
                self.in_call = False

        def close(self):
            self.closed_while_in_call = self.in_call
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
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(closing, 0.3)
        assert result.status is ToolResultStatus.TIMED_OUT
        assert time.monotonic() - started < 0.3
        assert client.closed == 0
        release_call.set()
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.01)
        assert client.closed == 1
        assert client.closed_while_in_call is False

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
            self.in_list = False
            self.closed_while_listing = False

        def list_tools(self):
            discovery_started.set()
            self.in_list = True
            try:
                release_discovery.wait()
                return []
            finally:
                self.in_list = False

        def close(self):
            self.closed_while_listing = self.in_list
            self.closed += 1

    async def scenario():
        client = BlockingDiscovery()
        gateway = _gateway(client, timeout_seconds=0.05)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.add_server("server", object()), 0.3)
        assert time.monotonic() - started < 0.3
        assert client.closed == 0
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.aclose(), 0.3)
        release_discovery.set()
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.01)
        assert client.closed == 1
        assert client.closed_while_listing is False

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


@pytest.mark.parametrize("idle_clients", [1, 3])
def test_idle_sync_close_has_capacity_when_operation_workers_are_exhausted(idle_clients) -> None:
    release_operations = threading.Event()
    operation_workers: list[threading.Thread] = []

    class SyncClose(Client):
        def aclose(self):
            self.closed += 1

    clients = [SyncClose() for _ in range(idle_clients)]
    late_clients = [SyncClose() for _ in range(4)]

    def factory(source):
        if source < idle_clients:
            return clients[source]
        operation_workers.append(threading.current_thread())
        release_operations.wait()
        return late_clients[source - idle_clients]

    async def scenario():
        gateway = _gateway(clients[0], timeout_seconds=0.1, factory=factory)
        for index in range(idle_clients):
            await gateway.add_server(f"idle-{index}", index)
        outcomes = await asyncio.gather(
            *(gateway.add_server(f"blocked-{index}", idle_clients + index) for index in range(4)),
            return_exceptions=True,
        )
        assert all(isinstance(outcome, TimeoutError) for outcome in outcomes)
        assert len(operation_workers) == 4
        assert all(worker.is_alive() for worker in operation_workers)
        # All four operations remain blocked until after shutdown and loop exit.
        await asyncio.wait_for(asyncio.gather(*(gateway.aclose() for _ in range(12))), 0.8)
        assert [client.closed for client in clients] == [1] * idle_clients
        await gateway.aclose()
        assert [client.closed for client in clients] == [1] * idle_clients
        assert all(worker.is_alive() for worker in operation_workers)

    try:
        asyncio.run(scenario())
    finally:
        release_operations.set()
        for worker in operation_workers:
            worker.join(1)
    assert [client.closed for client in late_clients] == [1] * 4


def test_cleanup_capacity_is_bounded_and_cannot_starve_operations() -> None:
    release_cleanup = threading.Event()
    cleanup_workers: list[threading.Thread] = []

    class BlockingClose(Client):
        def aclose(self):
            self.closed += 1
            cleanup_workers.append(threading.current_thread())
            release_cleanup.wait()

    clients = [BlockingClose() for _ in range(12)]
    healthy = Client()

    async def scenario():
        async def factory(source):
            return clients[source] if source < len(clients) else healthy

        gateway = _gateway(healthy, timeout_seconds=0.1, factory=factory)
        for index in range(len(clients)):
            await gateway.add_server(f"blocked-{index}", index)
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(gateway.remove_server(f"blocked-{index}") for index in range(len(clients))),
                return_exceptions=True,
            ),
            0.8,
        )
        assert all(isinstance(outcome, TimeoutError) for outcome in outcomes)
        assert len(cleanup_workers) == 4
        assert all(worker.is_alive() for worker in cleanup_workers)

        # A separate cleanup budget must leave ordinary synchronous work usable.
        def sync_factory(_source):
            return healthy

        gateway._factory = sync_factory
        await gateway.add_server("healthy", len(clients))
        result = await gateway.execute(_request(gateway))
        assert result.status is ToolResultStatus.SUCCESS
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(gateway.aclose() for _ in range(12)), return_exceptions=True),
            0.8,
        )
        assert all(isinstance(outcome, TimeoutError) for outcome in outcomes)
        assert all(outcome is outcomes[0] for outcome in outcomes)
        with pytest.raises(TimeoutError) as repeated:
            await gateway.aclose()
        assert repeated.value is outcomes[0]
        assert sum(client.closed for client in clients) == 4
        assert all(client.closed <= 1 for client in clients)
        assert healthy.closed == 1

    try:
        asyncio.run(scenario())
    finally:
        release_cleanup.set()
        for worker in cleanup_workers:
            worker.join(1)
    assert sum(client.closed for client in clients) == 4


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
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.aclose(), 0.3)
        assert client.closed == 0
        release_call.set()
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.01)
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
        assert len(workers) <= mcp_module._MAX_SYNC_CALLS_PER_GATEWAY
        assert len(created) <= mcp_module._MAX_SYNC_CALLS_PER_GATEWAY
        healthy = _gateway(Client(), timeout_seconds=0.05)
        await healthy.add_server("healthy", object())
        assert await asyncio.wait_for(healthy.aclose(), 0.2) is None
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


def test_blocked_async_factory_timeout_cancels_before_client_is_created() -> None:
    async def scenario():
        release_factory = asyncio.Event()
        client = Client()

        async def factory(_source):
            await release_factory.wait()
            return client

        gateway = _gateway(client, timeout_seconds=0.05, factory=factory)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.add_server("server", object()), 0.2)
        assert await asyncio.wait_for(gateway.aclose(), 0.2) is None
        assert client.closed == 0
        release_factory.set()
        await asyncio.sleep(0)
        assert client.closed == 0

    asyncio.run(scenario())


def test_blocked_async_discovery_never_closes_client_concurrently() -> None:
    class BlockingDiscovery(Client):
        def __init__(self):
            super().__init__()
            self.in_list = False
            self.closed_while_listing = False
            self.release_list = asyncio.Event()

        async def list_tools(self):
            self.in_list = True
            try:
                await self.release_list.wait()
                return []
            finally:
                self.in_list = False

        async def aclose(self):
            self.closed_while_listing = self.in_list
            await super().aclose()

    async def scenario():
        client = BlockingDiscovery()
        gateway = _gateway(client, timeout_seconds=0.05)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.add_server("server", object()), 0.2)
        for _ in range(100):
            if client.closed:
                break
            await asyncio.sleep(0.001)
        assert client.closed == 1
        assert await asyncio.wait_for(gateway.aclose(), 0.2) is None
        client.release_list.set()
        assert client.closed == 1
        assert client.closed_while_listing is False

    asyncio.run(scenario())


def test_blocked_async_call_times_out_without_concurrent_close() -> None:
    class BlockingCall(Client):
        def __init__(self):
            super().__init__()
            self.in_call = False
            self.closed_while_calling = False
            self.call_started = asyncio.Event()
            self.release_call = asyncio.Event()

        async def call_tool(self, _name, _arguments):
            self.in_call = True
            self.call_started.set()
            try:
                await self.release_call.wait()
                return {}
            finally:
                self.in_call = False

        async def aclose(self):
            self.closed_while_calling = self.in_call
            await super().aclose()

    async def scenario():
        client = BlockingCall()
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        executing = asyncio.create_task(gateway.execute(_request(gateway)))
        await client.call_started.wait()
        result = await asyncio.wait_for(executing, 0.2)
        assert result.status is ToolResultStatus.TIMED_OUT
        assert await asyncio.wait_for(gateway.aclose(), 0.2) is None
        assert client.closed == 1
        client.release_call.set()
        assert client.closed == 1
        assert client.closed_while_calling is False

    asyncio.run(scenario())


def test_blocked_async_close_has_bounded_idempotent_failure() -> None:
    class BlockingClose(Client):
        def __init__(self):
            super().__init__()
            self.close_calls = 0
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self):
            self.close_calls += 1
            self.close_started.set()
            await self.release_close.wait()
            self.closed += 1

    async def scenario():
        client = BlockingClose()
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.aclose(), 0.2)
        assert client.close_started.is_set()
        client.release_close.set()
        await asyncio.sleep(0)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.aclose(), 0.1)
        assert client.close_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("async_close", [False, True])
def test_abandoned_discovery_closes_after_original_loop_exits(async_close) -> None:
    release = threading.Event()
    closed = threading.Event()

    class Discovery:
        def __init__(self):
            self.close_count = 0
            self.listing = False

        def list_tools(self):
            self.listing = True
            try:
                release.wait()
                return []
            finally:
                self.listing = False

        def close(self):
            assert not self.listing
            self.close_count += 1
            closed.set()

    class AsyncCloseDiscovery(Discovery):
        async def aclose(self):
            await asyncio.sleep(0)
            self.close()

    client = AsyncCloseDiscovery() if async_close else Discovery()

    async def scenario():
        gateway = _gateway(client, timeout_seconds=0.05)
        with pytest.raises(TimeoutError):
            await gateway.add_server("server", object())
        with pytest.raises(TimeoutError):
            await gateway.aclose()
        assert client.close_count == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert closed.wait(1), "known client lost its cleanup owner when its loop exited"
    assert client.close_count == 1


def test_abandoned_sync_discovery_closes_async_client_on_live_original_loop() -> None:
    release = threading.Event()

    class Discovery(Client):
        def list_tools(self):
            release.wait()
            return []

        async def aclose(self):
            assert asyncio.get_running_loop() is self.origin
            await asyncio.sleep(0)
            await super().aclose()

    async def scenario():
        client = Discovery()
        client.origin = asyncio.get_running_loop()
        gateway = _gateway(client, timeout_seconds=0.05)
        with pytest.raises(TimeoutError):
            await gateway.add_server("server", object())
        release.set()
        await gateway.aclose()
        assert client.closed == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("shutdown", [False, True])
@pytest.mark.parametrize("async_list", [False, True])
def test_abandoned_refresh_recovers_only_live_server_after_settle(
    cancel, fails, shutdown, async_list
) -> None:
    release = threading.Event()
    started = threading.Event()

    class Refreshing(Client):
        def __init__(self):
            super().__init__()
            self.list_count = 0
            self.listing = False

        def list_tools(self):
            self.list_count += 1
            if self.list_count == 2:
                self.listing = True
                started.set()
                try:
                    release.wait()
                    if fails:
                        raise RuntimeError("late discovery failure")
                finally:
                    self.listing = False
            return [{"name": "read", "inputSchema": {"type": "object"}}]

        async def aclose(self):
            assert not self.listing
            await super().aclose()

    sync_list = Refreshing.list_tools
    if async_list:

        async def list_tools(self):
            if self.list_count == 1:
                self.listing = True
                started.set()
                try:
                    while not release.is_set():
                        await asyncio.sleep(0.001)
                finally:
                    self.listing = False
            return sync_list(self)

        Refreshing.list_tools = list_tools

    async def scenario():
        client = Refreshing()
        gateway = _gateway(client, timeout_seconds=0.2)
        await gateway.add_server("server", object())
        catalog = gateway._dynamic_catalog.specs
        generation = gateway._servers["server"].generation
        refresh = asyncio.create_task(gateway.refresh_server("server"))
        assert await asyncio.to_thread(started.wait, 1)
        if cancel:
            refresh.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await refresh
        if async_list:
            for _ in range(100):
                if not gateway._servers["server"].closing:
                    break
                await asyncio.sleep(0.001)
            assert not gateway._servers["server"].closing
        else:
            assert gateway._servers["server"].closing
        if shutdown:
            if async_list:
                assert await gateway.aclose() is None
            else:
                with pytest.raises(TimeoutError):
                    await gateway.aclose()
        release.set()
        for _ in range(100):
            if client.closed or (not shutdown and not gateway._servers["server"].closing):
                break
            await asyncio.sleep(0.005)
        if shutdown:
            assert gateway._dynamic_catalog.specs == ()
            assert "server" not in gateway._servers
            assert client.closed == 1
            if async_list:
                assert await gateway.aclose() is None
            else:
                with pytest.raises(TimeoutError):
                    await gateway.aclose()
        else:
            assert gateway._dynamic_catalog.specs == catalog
            assert await gateway._call("server", generation, "read", {}) == {
                "name": "read",
                "arguments": {},
            }
            if async_list and fails:
                with pytest.raises(RuntimeError, match="late discovery failure"):
                    await gateway.refresh_server("server")
            await gateway.refresh_server("server")
            removing = asyncio.create_task(gateway.remove_server("server"))
            outcomes = await asyncio.gather(removing, gateway.aclose(), return_exceptions=True)
            assert outcomes[0] is None or isinstance(outcomes[0], RuntimeError)
            assert outcomes[1] is None
            assert client.closed == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_abandoned_sync_discovery_returning_awaitable_preserves_cleanup_owner() -> None:
    release = threading.Event()

    class Discovery(Client):
        def list_tools(self):
            release.wait()
            return super().list_tools()

    async def scenario():
        client = Discovery()
        gateway = _gateway(client, timeout_seconds=0.05)
        with pytest.raises(TimeoutError):
            await gateway.add_server("server", object())
        await asyncio.sleep(0)
        release.set()
        await gateway.aclose()
        assert client.closed == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("operation", ["call", "refresh"])
@pytest.mark.parametrize("async_close", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_live_client_closes_after_timed_out_use_and_original_loop_exit(
    operation, async_close, fails
) -> None:
    release = threading.Event()
    closed = threading.Event()

    class Live:
        def __init__(self):
            self.lists = 0
            self.using = False
            self.close_count = 0

        def block(self):
            self.using = True
            try:
                release.wait()
                if fails:
                    raise RuntimeError("late failure")
            finally:
                self.using = False

        def list_tools(self):
            self.lists += 1
            if self.lists > 1:
                self.block()
            return [{"name": "read", "inputSchema": {"type": "object"}}]

        def call_tool(self, _name, _arguments):
            self.block()
            return {}

        def close(self):
            assert not self.using
            self.close_count += 1
            closed.set()

    class AsyncCloseLive(Live):
        async def aclose(self):
            await asyncio.sleep(0)
            self.close()

    client = AsyncCloseLive() if async_close else Live()

    async def scenario():
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        with pytest.raises(TimeoutError):
            if operation == "call":
                await gateway._call("server", generation, "read", {})
            else:
                await gateway.refresh_server("server")
        with pytest.raises(TimeoutError):
            await gateway.aclose()
        assert client.close_count == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert closed.wait(1), "live client cleanup depended on the closed loop"
    assert client.close_count == 1


def test_closed_loop_affine_client_close_failure_is_not_retried() -> None:
    release = threading.Event()
    attempted = threading.Event()

    class LoopAffine(Client):
        def call_tool(self, _name, _arguments):
            release.wait()
            return {}

        async def aclose(self):
            self.closed += 1
            attempted.set()
            if asyncio.get_running_loop() is not self.origin:
                raise RuntimeError("client requires its destroyed original loop")

    client = LoopAffine()
    gateway = _gateway(client, timeout_seconds=0.05)

    async def scenario():
        client.origin = asyncio.get_running_loop()
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        with pytest.raises(TimeoutError):
            await gateway._call("server", generation, "read", {})
        with pytest.raises(TimeoutError):
            await gateway.aclose()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert attempted.wait(1)
    with pytest.raises(TimeoutError):
        asyncio.run(gateway.aclose())
    assert client.closed == 1


def test_closed_loop_cleanup_waits_for_all_underlying_calls() -> None:
    releases = [threading.Event(), threading.Event()]
    finished = [threading.Event(), threading.Event()]
    closed = threading.Event()

    class Concurrent(Client):
        def call_tool(self, _name, arguments):
            index = arguments["index"]
            releases[index].wait()
            finished[index].set()
            return {}

        async def aclose(self):
            assert all(event.is_set() for event in finished)
            await super().aclose()
            closed.set()

    client = Concurrent()

    async def scenario():
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        outcomes = await asyncio.gather(
            *(gateway._call("server", generation, "read", {"index": i}) for i in range(2)),
            return_exceptions=True,
        )
        assert all(isinstance(outcome, TimeoutError) for outcome in outcomes)
        with pytest.raises(TimeoutError):
            await gateway.aclose()

    try:
        asyncio.run(scenario())
        releases[0].set()
        assert finished[0].wait(1)
        assert not closed.wait(0.05)
    finally:
        for event in releases:
            event.set()
    assert closed.wait(1)
    assert client.closed == 1


def test_remote_annotations_cannot_remove_approval_and_identity_is_stable() -> None:
    async def scenario():
        first = _gateway(Client())
        second = _gateway(Client())
        await first.add_server("server", object())
        await second.add_server("server", object())
        first_spec = first._dynamic_catalog.specs[0]
        second_spec = second._dynamic_catalog.specs[0]
        assert first_spec.name == "server__read"
        assert first_spec.requires_approval is True
        assert first_spec.workspace_id == second_spec.workspace_id
        before = first_spec.workspace_id
        await first.refresh_server("server")
        assert first._dynamic_catalog.specs[0].workspace_id == before
        await first.aclose()
        await second.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("remote_name", ["bad.name", "bad/name", "空白", "a" * 200])
def test_provider_unsafe_remote_names_are_normalized_and_call_original(remote_name: str) -> None:
    class Unsafe(Client):
        def __init__(self):
            super().__init__()
            self.called_name = None

        async def list_tools(self):
            return [{"name": remote_name, "inputSchema": {"type": "object"}}]

        async def call_tool(self, name, arguments):
            self.called_name = name
            return {"name": name, "arguments": arguments}

    async def scenario():
        client = Unsafe()
        gateway = _gateway(client)
        await gateway.add_server("server", object())
        public_name = gateway._dynamic_catalog.specs[0].name
        assert public_name.startswith("server__")
        assert len(public_name.encode("ascii")) <= 64
        assert mcp_module._PROVIDER_NAME.fullmatch(public_name) is not None
        result = await gateway.execute(_request(gateway))
        assert result.status is ToolResultStatus.SUCCESS
        assert client.called_name == remote_name
        await gateway.aclose()

    asyncio.run(scenario())


def test_optional_none_description_is_normalized_to_empty_text() -> None:
    class OptionalDescription(Client):
        async def list_tools(self):
            return [{"name": "read", "description": None, "inputSchema": {}}]

    async def scenario():
        gateway = _gateway(OptionalDescription())
        await gateway.add_server("server", object())
        assert gateway._dynamic_catalog.specs[0].description == ""
        await gateway.aclose()

    asyncio.run(scenario())


def test_safe_trailing_underscore_names_remain_distinct() -> None:
    class Distinct(Client):
        async def list_tools(self):
            return [
                {"name": "read", "inputSchema": {}},
                {"name": "read_", "inputSchema": {}},
            ]

    async def scenario():
        gateway = _gateway(Distinct())
        await gateway.add_server("server", object())
        assert [spec.name for spec in gateway._dynamic_catalog.specs] == [
            "server__read",
            "server__read_",
        ]
        await gateway.aclose()

    asyncio.run(scenario())


def test_long_server_name_accepts_a_short_tool_name() -> None:
    class Short(Client):
        async def list_tools(self):
            return [{"name": "x", "inputSchema": {}}]

    async def scenario():
        gateway = _gateway(Short())
        await gateway.add_server("s" * 60, object())
        assert gateway._dynamic_catalog.specs[0].name == f"{'s' * 60}__x"
        await gateway.aclose()

    asyncio.run(scenario())


def test_remote_name_normalization_collision_is_rejected() -> None:
    class Colliding(Client):
        async def list_tools(self):
            return [
                {"name": "a.b", "inputSchema": {}},
                {"name": "a/b", "inputSchema": {}},
            ]

    async def scenario():
        gateway = _gateway(Colliding())
        with pytest.raises(ValueError, match="unique"):
            await gateway.add_server("server", object())
        await gateway.aclose()

    asyncio.run(scenario())


def test_async_context_manager_client_is_entered_and_exited_once() -> None:
    client = Client()

    class Manager:
        def __init__(self):
            self.entered = 0
            self.exited = 0

        async def __aenter__(self):
            self.entered += 1
            return client

        async def __aexit__(self, exc_type, exc, traceback):
            self.exited += 1

    manager = Manager()

    async def scenario():
        gateway = _gateway(client, factory=lambda _source: manager)
        await gateway.add_server("server", object())
        await gateway.aclose()

    asyncio.run(scenario())
    assert manager.entered == manager.exited == 1
    assert client.closed == 0


def test_async_context_manager_owns_enter_calls_and_exit_in_one_task() -> None:
    owner = None
    entered_group = None
    calls = []

    class ScopedClient(Client):
        async def list_tools(self):
            calls.append(("list", asyncio.current_task()))
            assert asyncio.current_task() is owner
            return await super().list_tools()

        async def call_tool(self, name, arguments):
            calls.append(("call", asyncio.current_task()))
            assert asyncio.current_task() is owner
            return await super().call_tool(name, arguments)

    client = ScopedClient()

    class Manager:
        async def __aenter__(self):
            nonlocal owner, entered_group
            owner = asyncio.current_task()
            entered_group = anyio.create_task_group()
            await entered_group.__aenter__()
            return client

        async def __aexit__(self, exc_type, exc, traceback):
            calls.append(("exit", asyncio.current_task()))
            assert asyncio.current_task() is owner
            assert entered_group is not None
            await entered_group.__aexit__(exc_type, exc, traceback)

    async def scenario():
        gateway = _gateway(client, factory=lambda _source: Manager())
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        await gateway._call("server", generation, "read", {})
        await gateway.remove_server("server")

    asyncio.run(scenario())
    assert [kind for kind, _task in calls] == ["list", "call", "exit"]
    assert {task for _kind, task in calls} == {owner}


def test_cancelling_a_queued_async_context_call_does_not_cancel_the_active_call() -> None:
    class Blocking(Client):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def call_tool(self, name, arguments):
            self.started.set()
            await self.release.wait()
            return {"name": name, "arguments": arguments}

    async def scenario():
        client = Blocking()

        class Manager:
            async def __aenter__(self):
                return client

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        gateway = _gateway(client, factory=lambda _source: Manager())
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        active = asyncio.create_task(gateway._call("server", generation, "read", {"id": 1}))
        await client.started.wait()
        queued = asyncio.create_task(gateway._call("server", generation, "read", {"id": 2}))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert not active.done()
        client.release.set()
        assert await active == {"name": "read", "arguments": {"id": 1}}
        assert gateway._dynamic_catalog.specs
        await gateway.aclose()

    asyncio.run(scenario())


def test_late_caller_cancellation_after_result_does_not_stop_owner() -> None:
    caller = None

    class RacingClient:
        async def call_tool(self, value):
            if value == "race":
                assert caller is not None
                asyncio.get_running_loop().call_soon(caller.cancel)
            return value

    class Manager:
        async def __aenter__(self):
            return RacingClient()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def scenario():
        nonlocal caller
        proxy = await mcp_module._AsyncContextClient(
            Manager(), 0.2, mcp_module._SyncLimiter()
        ).start(0.2)
        caller = asyncio.current_task()
        assert caller is not None
        with pytest.raises(asyncio.CancelledError):
            await proxy.call_tool("race")
        while caller.cancelling():
            caller.uncancel()

        assert await proxy.call_tool("next") == "next"
        await proxy.aclose()

    asyncio.run(scenario())


def test_async_context_method_cancellation_removes_the_dead_catalog() -> None:
    class Cancelled(Client):
        async def call_tool(self, name, arguments):
            raise asyncio.CancelledError

    class Manager:
        async def __aenter__(self):
            return client

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def scenario():
        gateway = _gateway(client, factory=lambda _source: Manager())
        await gateway.add_server("server", object())
        generation = gateway._servers["server"].generation
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(gateway._call("server", generation, "read", {}), 0.2)
        for _ in range(20):
            if not gateway._dynamic_catalog.specs:
                break
            await asyncio.sleep(0)
        assert gateway._dynamic_catalog.specs == ()
        await gateway.aclose()

    client = Cancelled()
    asyncio.run(scenario())


def test_async_context_enter_timeout_is_bounded_when_cancellation_is_suppressed() -> None:
    exited = asyncio.Event()

    class Manager:
        async def __aenter__(self):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return Client()

        async def __aexit__(self, exc_type, exc, traceback):
            exited.set()

    async def scenario():
        gateway = _gateway(Client(), timeout_seconds=0.03, factory=lambda _source: Manager())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.add_server("server", object()), 0.2)
        await asyncio.wait_for(exited.wait(), 0.2)
        await gateway.aclose()

    asyncio.run(scenario())


def test_sync_client_from_async_context_does_not_block_the_event_loop() -> None:
    started = threading.Event()
    release = threading.Event()

    class SyncClient:
        def list_tools(self):
            started.set()
            release.wait()
            return [{"name": "read", "inputSchema": {}}]

        def close(self):
            return None

    class Manager:
        async def __aenter__(self):
            return SyncClient()

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def scenario():
        gateway = _gateway(Client(), factory=lambda _source: Manager())
        before = time.monotonic()
        adding = asyncio.create_task(gateway.add_server("server", object()))
        assert await asyncio.to_thread(started.wait, 1)
        assert time.monotonic() - before < 0.1
        ticked = asyncio.Event()
        asyncio.get_running_loop().call_soon(ticked.set)
        await asyncio.wait_for(ticked.wait(), 0.05)
        release.set()
        await adding
        await gateway.aclose()

    watchdog = threading.Timer(0.2, release.set)
    watchdog.start()
    try:
        asyncio.run(scenario())
    finally:
        release.set()
        watchdog.cancel()


def test_async_timeout_cancels_underlying_mcp_call() -> None:
    class Blocking(Client):
        def __init__(self):
            super().__init__()
            self.cancelled = asyncio.Event()

        async def call_tool(self, name, arguments):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    async def scenario():
        client = Blocking()
        gateway = _gateway(client, timeout_seconds=0.05)
        await gateway.add_server("server", object())
        result = await gateway.execute(_request(gateway))
        assert result.status is ToolResultStatus.TIMED_OUT
        await asyncio.wait_for(client.cancelled.wait(), 0.2)
        await gateway.aclose()

    asyncio.run(scenario())


def test_async_context_timeout_cancels_only_the_current_call() -> None:
    class StatefulClient(Client):
        def __init__(self):
            super().__init__()
            self.cancelled = asyncio.Event()

        async def list_tools(self):
            return [
                {"name": "slow", "inputSchema": {"type": "object"}},
                {"name": "fast", "inputSchema": {"type": "object"}},
            ]

        async def call_tool(self, name, arguments):
            if name == "fast":
                return {"name": name, "arguments": arguments}
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class Manager:
        async def __aenter__(self):
            self.group = anyio.create_task_group()
            await self.group.__aenter__()
            return client

        async def __aexit__(self, exc_type, exc, traceback):
            await self.group.__aexit__(exc_type, exc, traceback)

    async def scenario():
        gateway = _gateway(client, timeout_seconds=0.05, factory=lambda _source: Manager())
        await gateway.add_server("server", object())

        timed_out = await gateway.execute(_request(gateway))
        assert timed_out.status is ToolResultStatus.TIMED_OUT
        await asyncio.wait_for(client.cancelled.wait(), 0.2)
        await asyncio.sleep(0)
        assert [spec.name for spec in gateway._dynamic_catalog.specs] == [
            "server__slow",
            "server__fast",
        ]

        succeeded = await gateway.execute(
            _request(gateway, spec_index=1, call_id="call-2", step_id="step-2")
        )
        assert succeeded.status is ToolResultStatus.SUCCESS
        await gateway.remove_server("server")
        await gateway.aclose()

    client = StatefulClient()
    asyncio.run(scenario())


def test_mcp_error_result_maps_to_failed_tool_result() -> None:
    class ErrorResult(Client):
        async def call_tool(self, name, arguments):
            return {"isError": True, "content": [{"type": "text", "text": "failed"}]}

    async def scenario():
        gateway = _gateway(ErrorResult())
        await gateway.add_server("server", object())
        result = await gateway.execute(_request(gateway))
        assert result.status is ToolResultStatus.FAILED
        await gateway.aclose()

    asyncio.run(scenario())


def test_mcp_model_result_is_normalized_to_json_envelope() -> None:
    @dataclass
    class TextContent:
        type: str
        text: str

    class CallToolResult:
        isError = False
        structuredContent = {"answer": 42}

        def __init__(self):
            self.content = [TextContent("text", "hello")]

    class ModelResult(Client):
        async def call_tool(self, name, arguments):
            return CallToolResult()

    async def scenario():
        gateway = _gateway(ModelResult())
        await gateway.add_server("server", object())
        result = await gateway.execute(_request(gateway))
        assert result.status is ToolResultStatus.SUCCESS
        assert result.inline_content is not None
        assert result.inline_content.value == {
            "content": [{"type": "text", "text": "hello"}],
            "structuredContent": {"answer": 42},
            "isError": False,
        }
        await gateway.aclose()

    asyncio.run(scenario())
