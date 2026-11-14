import asyncio
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent
from src.core.tool import RemoteToolError
from src.mcp.mcp_client import MCPManager
from src.schema.tool import Tool
from src.tool.tool_registry import DefaultToolRegistry


def _remote(name, description="Remote tool"):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )


class FakeClient:
    def __init__(self, tools=(), *, enter_error=None, list_error=None, close_error=None):
        self.tools = list(tools)
        self.enter_error = enter_error
        self.list_error = list_error
        self.close_error = close_error
        self.entered = False
        self.closed = False
        self.calls = []
        self.result = CallToolResult(content=[TextContent(text="ok")])
        self.call_started = asyncio.Event()
        self.call_release = asyncio.Event()
        self.block_calls = False

    async def __aenter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        self.entered = True
        return self

    async def __aexit__(self, *_args):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error

    async def list_tools(self):
        if self.list_error is not None:
            raise self.list_error
        return list(self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        self.call_started.set()
        if self.block_calls:
            await self.call_release.wait()
        return self.result


class ClientFactory:
    def __init__(self, *clients):
        self.clients = list(clients)
        self.calls = []

    def __call__(self, source, **options):
        self.calls.append((source, options))
        return self.clients.pop(0)


class FailingRegistrationCatalog(DefaultToolRegistry):
    def register_many_strict(self, tools):
        raise RuntimeError("registration failed")


def test_add_server_atomically_registers_and_calls_remote_tools():
    async def scenario():
        client = FakeClient([_remote("forecast")])
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))

        await manager.add_server("weather", "memory://weather")
        tool = catalog.find_tool("mcp__weather__forecast")
        assert tool is not None
        output = await tool.executer({"city": "Beijing"})

        assert output["content"] == [{"type": "text", "text": "ok"}]
        assert client.calls == [("forecast", {"city": "Beijing"})]
        assert manager.catalog is catalog

    asyncio.run(scenario())


def test_add_server_rejects_duplicate_names_without_opening_another_client():
    async def scenario():
        first = FakeClient([_remote("one")])
        second = FakeClient([_remote("two")])
        factory = ClientFactory(first, second)
        manager = MCPManager(DefaultToolRegistry(), client_factory=factory)
        await manager.add_server("server", "first")

        with pytest.raises(ValueError, match="already connected"):
            await manager.add_server("server", "second")

        assert len(factory.calls) == 1
        assert second.entered is False

    asyncio.run(scenario())


def test_add_server_catalog_collision_closes_candidate_and_leaves_catalog_unchanged():
    async def scenario():
        catalog = DefaultToolRegistry()
        existing = Tool("mcp__weather__forecast", "existing", "Existing.", {}, lambda x: x)
        catalog.register(existing)
        client = FakeClient([_remote("forecast")])
        manager = MCPManager(catalog, client_factory=ClientFactory(client))

        with pytest.raises(ValueError, match="collision"):
            await manager.add_server("weather", "source")

        assert catalog.get_all_tools() == [existing]
        assert client.closed is True

    asyncio.run(scenario())


def test_add_server_normalized_name_collision_registers_nothing():
    async def scenario():
        catalog = DefaultToolRegistry()
        client = FakeClient([_remote("daily forecast"), _remote("daily_forecast")])
        manager = MCPManager(catalog, client_factory=ClientFactory(client))

        with pytest.raises(ValueError, match="collision"):
            await manager.add_server("weather", "source")

        assert catalog.get_all_tools() == []
        assert client.closed is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeClient(enter_error=RuntimeError("connect failed")), "connect failed"),
        (FakeClient(list_error=RuntimeError("list failed")), "list failed"),
    ],
)
def test_failed_connect_or_list_leaves_no_server_or_tools(client, message):
    async def scenario():
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))

        with pytest.raises(RuntimeError, match=message):
            await manager.add_server("server", "source")

        assert catalog.get_all_tools() == []
        with pytest.raises(KeyError):
            await manager.remove_server("server")

    asyncio.run(scenario())


def test_failed_registration_closes_candidate_and_leaves_no_server_or_tools():
    async def scenario():
        client = FakeClient([_remote("one")])
        catalog = FailingRegistrationCatalog()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))

        with pytest.raises(RuntimeError, match="registration failed"):
            await manager.add_server("server", "source")

        assert catalog.get_all_tools() == []
        assert client.closed is True
        with pytest.raises(KeyError, match="not connected"):
            await manager.remove_server("server")

    asyncio.run(scenario())


def test_refresh_replaces_only_that_servers_tools():
    async def scenario():
        client = FakeClient([_remote("old")])
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))
        await manager.add_server("server", "source")
        old = catalog.find_tool("mcp__server__old")
        client.tools = [_remote("new")]

        await manager.refresh_server("server")

        assert catalog.find_tool("mcp__server__old") is None
        assert catalog.find_tool("mcp__server__new") is not None
        with pytest.raises(KeyError, match="not connected"):
            await old.executer({})

    asyncio.run(scenario())


def test_refresh_collision_rolls_back_to_the_previous_tools():
    async def scenario():
        client = FakeClient([_remote("old")])
        catalog = DefaultToolRegistry()
        catalog.register(Tool("taken", "mcp__server__new", "Taken.", {}, lambda x: x))
        manager = MCPManager(catalog, client_factory=ClientFactory(client))
        await manager.add_server("server", "source")
        client.tools = [_remote("new")]

        with pytest.raises(ValueError, match="collision"):
            await manager.refresh_server("server")

        assert catalog.find_tool("mcp__server__old") is not None
        assert catalog.find_tool("mcp__server__new").id == "taken"

    asyncio.run(scenario())


def test_remove_unregisters_tools_and_stale_tool_fails():
    async def scenario():
        client = FakeClient([_remote("one")])
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))
        await manager.add_server("server", "source")
        stale = catalog.find_tool("mcp__server__one")

        await manager.remove_server("server")

        assert catalog.find_tool("mcp__server__one") is None
        assert client.closed is True
        with pytest.raises(KeyError, match="not connected"):
            await stale.executer({})

    asyncio.run(scenario())


def test_remove_clears_state_even_when_client_close_fails():
    async def scenario():
        client = FakeClient([_remote("one")], close_error=RuntimeError("close failed"))
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))
        await manager.add_server("server", "source")
        stale = catalog.find_tool("mcp__server__one")

        with pytest.raises(RuntimeError, match="close failed"):
            await manager.remove_server("server")

        assert catalog.get_all_tools() == []
        with pytest.raises(KeyError, match="not connected"):
            await stale.executer({})
        with pytest.raises(KeyError, match="not connected"):
            await manager.remove_server("server")

    asyncio.run(scenario())


def test_aclose_is_idempotent_and_closes_every_server():
    async def scenario():
        first = FakeClient([_remote("one")])
        second = FakeClient([_remote("two")])
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(first, second))
        await manager.add_server("first", "one")
        await manager.add_server("second", "two")

        await manager.aclose()
        await manager.aclose()

        assert first.closed is second.closed is True
        assert catalog.get_all_tools() == []
        with pytest.raises(RuntimeError, match="closed"):
            await manager.add_server("third", "three")

    asyncio.run(scenario())


def test_remove_waits_for_an_inflight_call_but_rejects_new_calls_immediately():
    async def scenario():
        client = FakeClient([_remote("one")])
        client.block_calls = True
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))
        await manager.add_server("server", "source")
        stale = catalog.find_tool("mcp__server__one")
        running = asyncio.create_task(stale.executer({"call": 1}))
        await client.call_started.wait()

        removing = asyncio.create_task(manager.remove_server("server"))
        await asyncio.sleep(0)
        assert catalog.find_tool("mcp__server__one") is None
        with pytest.raises(KeyError, match="not connected"):
            await stale.executer({"call": 2})
        assert removing.done() is False
        assert client.closed is False

        client.call_release.set()
        assert (await running)["is_error"] is False
        await removing
        assert client.closed is True

    asyncio.run(scenario())


def test_transport_failure_becomes_a_safe_remote_error():
    async def scenario():
        client = FakeClient([_remote("one")])

        async def fail(_name, _arguments):
            raise RuntimeError("transport offline")

        client.call_tool = fail
        catalog = DefaultToolRegistry()
        manager = MCPManager(catalog, client_factory=ClientFactory(client))
        await manager.add_server("server", "source")
        tool = catalog.find_tool("mcp__server__one")

        with pytest.raises(RemoteToolError) as caught:
            await tool.executer({})

        assert caught.value.output["is_error"] is True
        assert "transport offline" in caught.value.output["structured_content"]["error"]

    asyncio.run(scenario())
