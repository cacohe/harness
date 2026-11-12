from src.core.tool import ToolRegistry
from src.schema.tool import Tool


class DefaultToolRegistry(ToolRegistry):
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.id] = tool

    def unregister(self, tool_id: str):
        self._tools.pop(tool_id)

    def get_tool(self, tool_id: str) -> Tool:
        return self._tools[tool_id]

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())
