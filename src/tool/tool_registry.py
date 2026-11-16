import hashlib
import json

from src.schema.tool import Tool


class DefaultToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.id] = tool

    def register_many_strict(self, tools: list[Tool]) -> None:
        """Register a batch only when every id and name is unique."""
        snapshot = dict(self._tools)
        batch_ids: set[str] = set()
        batch_names: set[str] = set()
        existing_names = {tool.name for tool in self._tools.values()}
        for tool in tools:
            if (
                tool.id in self._tools
                or tool.id in batch_ids
                or tool.name in existing_names
                or tool.name in batch_names
            ):
                raise ValueError(f"tool collision for id {tool.id!r} or name {tool.name!r}")
            batch_ids.add(tool.id)
            batch_names.add(tool.name)
        try:
            for tool in tools:
                self.register(tool)
        except BaseException:
            self._tools = snapshot
            raise

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

    def capability_contracts(self) -> list[dict]:
        """Return deterministic public contracts without callables or workspace paths."""
        return sorted(
            [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "origin": tool.origin.value,
                    "requires_approval": tool.requires_approval,
                    "requires_sandbox": tool.requires_sandbox,
                    "sandbox_kind": (
                        None if tool.sandbox_kind is None else tool.sandbox_kind.value
                    ),
                }
                for tool in self._tools.values()
            ],
            key=lambda item: (item["id"], item["name"]),
        )

    def capability_digest(self) -> str:
        encoded = json.dumps(
            self.capability_contracts(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
