from src.schema.tool import Tool


class DefaultToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.id] = tool

    def unregister(self, tool_id: str):
        self._tools.pop(tool_id)

    def get_tool(self, tool_id: str) -> Tool:
        return self._tools[tool_id]

    def find_tool(self, name: str) -> Tool | None:
        return next(
            (tool for tool in self._tools.values() if tool.name == name or tool.id == name),
            None,
        )

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())
