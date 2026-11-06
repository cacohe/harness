from contextlib import AsyncExitStack
from typing import Any

from fastmcp import Client

from src.schema.tool import Tool


class MCPManager:
    """Manage connected MCP servers and the tools they provide."""

    def __init__(self):
        """Create a manager with no servers connected."""
        self._clients: dict[str, Client] = {}
        self._stacks: dict[str, AsyncExitStack] = {}

    async def add_mcp_server(self, name: str, source: Any, **options) -> None:
        """
        Connect one MCP server and keep it available.
        Args:
            name: Server name. Must be a non-empty string without '/'.
            source: Connection source. A URL, a script path, or a dict with url or command.
            **options: Options forwarded to the FastMCP client.
        Raises:
            ValueError: If the name is invalid or the server is already connected.
        """
        self._validate_name(name)
        if name in self._clients:
            raise ValueError(f"MCP server {name!r} is already connected")

        client = Client(self._client_source(name, source), **options)
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            await stack.enter_async_context(client)
        except BaseException:
            await stack.aclose()
            raise
        self._clients[name] = client
        self._stacks[name] = stack

    async def remove_mcp_server(self, name: str) -> None:
        """
        Disconnect one MCP server and drop its tools.
        Args:
            name: Name passed to add_mcp_server.
        Raises:
            KeyError: If that server is not connected.
        """
        if name not in self._clients:
            raise KeyError(f"MCP server {name!r} is not connected")
        self._clients.pop(name)
        stack = self._stacks.pop(name)
        await stack.aclose()

    async def list_tools(self) -> list[Tool]:
        """
        List every tool provided by the connected MCP servers.
        Returns:
            Tools from all connected servers. Call them through call_tool.
        """
        tools: list[Tool] = []
        for server_name, client in list(self._clients.items()):
            remote_tools = await client.list_tools()
            tools.extend(self._to_tool(server_name, remote) for remote in remote_tools)
        return tools

    async def call_tool(self, tool: Tool, arguments: dict | None = None) -> Any:
        """
        Call a tool returned by list_tools.
        Args:
            tool: Tool from list_tools.
            arguments: Arguments passed to the remote tool.
        Returns:
            The tool result data returned by FastMCP.
        """
        return await tool.executer(arguments)

    @staticmethod
    def _validate_name(name: str) -> None:
        """
        Reject a server name that cannot be used in a tool id.
        Args:
            name: Proposed server name.
        Raises:
            ValueError: If the name is empty or contains '/'.
        """
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"MCP server name must be a non-empty string without '/', got {name!r}"
            )

    @staticmethod
    def _client_source(name: str, source: Any) -> Any:
        """
        Wrap a url or command config so FastMCP can open that server.
        Args:
            name: Server name used inside a single-server config.
            source: Connection source passed for this server.
        Returns:
            A FastMCP client source.
        """
        if (
            isinstance(source, dict)
            and "mcpServers" not in source
            and ("url" in source or "command" in source)
        ):
            return {"mcpServers": {name: source}}
        return source

    def _to_tool(self, server_name: str, remote) -> Tool:
        """
        Convert one remote MCP tool into a local Tool.
        Args:
            server_name: Name of the server that advertised the tool.
            remote: Tool object returned by the server's list_tools.
        Returns:
            A local tool whose executor calls the remote tool.
        """
        remote_name = remote.name
        tool_id = f"{server_name}/{remote_name}"
        schema = getattr(remote, "input_schema", None)
        if schema is None:
            schema = getattr(remote, "inputSchema", None) or {}
        description = f"[{server_name}] {remote.description or ''}".rstrip()

        async def execute(arguments: dict | None = None, **kwargs):
            """
            Call this remote tool with the given arguments.
            Args:
                arguments: Argument mapping for the remote tool.
                **kwargs: Extra arguments merged into that mapping.
            Returns:
                The tool result data returned by FastMCP.
            Raises:
                KeyError: If the server has been removed.
            """
            client = self._clients.get(server_name)
            if client is None:
                raise KeyError(f"MCP server {server_name!r} is not connected")
            payload = dict(arguments or {})
            payload.update(kwargs)
            result = await client.call_tool(remote_name, payload)
            return result.data

        return Tool(
            id=tool_id,
            name=tool_id,
            description=description,
            input_schema=dict(schema),
            executer=execute,
        )
