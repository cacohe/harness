from src.core.tool import ToolRegistry
from src.schema.tool import Tool


class DefaultToolRegistry(ToolRegistry):
    tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.id] = tool

    def unregister(self, tool_id: str):
        self.tools.pop(tool_id)

    def get_tool(self, tool_id: str) -> Tool:
        return self.tools[tool_id]

    def get_all_tools(self) -> list[Tool]:
        return list(self.tools.values())
