import asyncio

from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.tools import InMemoryToolExecutionStore
from src.adapters.tools.mcp import MCPToolGateway


class Client:
    def __init__(self):
        self.closed = 0

    async def list_tools(self):
        return [{"name": "read", "description": "read", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}}]

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


def _gateway(client):
    return MCPToolGateway(
        InMemoryToolExecutionStore(),
        PolicyArtifactExternalizer(
            InMemoryArtifactRepository(1024),
            ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
        ),
        client_factory=lambda _source: client,
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
