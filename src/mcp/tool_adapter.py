"""Convert FastMCP tool definitions and results into harness runtime values."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from src.core.tool import RemoteToolError
from src.schema.tool import Tool, ToolOrigin
from src.utils.tool_names import provider_tool_name as _provider_tool_name


def provider_tool_name(server_name: str, remote_name: str) -> str:
    """Build one deterministic provider-safe function name."""
    return _provider_tool_name("mcp", server_name, remote_name)


def normalize_mcp_result(result: Any) -> dict[str, Any]:
    """Return the stable JSON-safe envelope persisted for one MCP call."""
    try:
        content = [_content_block(block) for block in list(getattr(result, "content", []))]
        structured = getattr(result, "structured_content", None)
        is_error = bool(getattr(result, "is_error", False))
        envelope = {
            "content": content,
            "structured_content": structured,
            "is_error": is_error,
        }
        _validate_strict_json(envelope)
        json.dumps(envelope, ensure_ascii=False, allow_nan=False)
    except Exception as error:
        if isinstance(error, RemoteToolError):
            raise
        raise RemoteToolError(_invalid_result_envelope(error)) from error
    if is_error:
        raise RemoteToolError(envelope)
    return envelope


def _validate_strict_json(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MCP results require finite numbers")
        return
    if isinstance(value, list):
        for item in value:
            _validate_strict_json(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("MCP result object keys must be strings")
        for item in value.values():
            _validate_strict_json(item)
        return
    raise TypeError(f"unsupported MCP JSON value: {type(value).__name__}")


def _content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, BaseModel):
        return block.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(block, dict):
        return dict(block)
    raise TypeError(f"unsupported MCP content block: {type(block).__name__}")


def _invalid_result_envelope(error: Exception) -> dict[str, Any]:
    return {
        "content": [],
        "structured_content": {"error": f"Invalid MCP result: {type(error).__name__}: {error}"},
        "is_error": True,
    }


def make_mcp_tool(
    server_name: str,
    remote_tool: Any,
    invoke: Callable[[str, dict[str, Any]], Awaitable[Any]],
) -> Tool:
    """Create an approval-gated harness tool for one remote MCP definition."""
    remote_name = str(remote_tool.name)
    tool_name = provider_tool_name(server_name, remote_name)
    schema = getattr(remote_tool, "input_schema", None)
    if schema is None:
        schema = getattr(remote_tool, "inputSchema", None) or {}

    async def execute(arguments: dict | None = None, **kwargs: Any) -> dict[str, Any]:
        payload = dict(arguments or {})
        payload.update(kwargs)
        result = await invoke(remote_name, payload)
        return normalize_mcp_result(result)

    description = f"[MCP {server_name}/{remote_name}] {remote_tool.description or ''}".rstrip()
    return Tool(
        id=tool_name,
        name=tool_name,
        description=description,
        input_schema=dict(schema),
        executer=execute,
        origin=ToolOrigin.MCP,
        requires_approval=True,
    )
