import asyncio

import pytest
from src.mcp.mcp_client import MCPManager


def test_add_mcp_server_rejects_a_name_containing_a_slash():
    async def scenario():
        manager = MCPManager()
        with pytest.raises(ValueError, match="without '/'"):
            await manager.add_mcp_server("weather/local", "https://example.com/mcp")

    asyncio.run(scenario())


def test_add_mcp_server_rejects_an_empty_name():
    async def scenario():
        manager = MCPManager()
        with pytest.raises(ValueError, match="non-empty"):
            await manager.add_mcp_server("", "https://example.com/mcp")

    asyncio.run(scenario())


def test_remove_mcp_server_rejects_an_unknown_server():
    async def scenario():
        manager = MCPManager()
        with pytest.raises(KeyError, match="missing"):
            await manager.remove_mcp_server("missing")

    asyncio.run(scenario())
