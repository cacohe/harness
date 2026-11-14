import asyncio
import math
import re

import pytest
from mcp.types import CallToolResult, TextContent
from src.core.tool import RemoteToolError
from src.mcp.tool_adapter import make_mcp_tool, normalize_mcp_result, provider_tool_name
from src.schema.tool import ToolOrigin


def test_provider_tool_name_uses_a_stable_mcp_namespace():
    assert provider_tool_name("weather", "forecast") == "mcp__weather__forecast"


def test_provider_tool_name_normalizes_to_provider_safe_ascii():
    name = provider_tool_name("weather api", "daily.forecast/v2")

    assert name == "mcp__weather_api__daily_forecast_v2"
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)


def test_provider_tool_name_truncates_deterministically_with_a_hash_suffix():
    first = provider_tool_name("server", "forecast_" + "x" * 100)
    repeated = provider_tool_name("server", "forecast_" + "x" * 100)
    different = provider_tool_name("server", "forecast_" + "y" * 100)

    assert first == repeated
    assert len(first) == 64
    assert first != different


@pytest.mark.parametrize(("server", "tool"), [("天气", "forecast"), ("weather", "!!!")])
def test_provider_tool_name_rejects_an_empty_normalized_segment(server, tool):
    with pytest.raises(ValueError, match="provider-safe"):
        provider_tool_name(server, tool)


def test_provider_tool_name_exposes_normalization_collisions_to_the_manager():
    assert provider_tool_name("weather api", "forecast") == provider_tool_name(
        "weather_api", "forecast"
    )


def test_normalize_mcp_result_returns_a_stable_json_envelope():
    result = CallToolResult(
        content=[TextContent(text="sunny")],
        structuredContent={"temperature": 18},
    )

    envelope = normalize_mcp_result(result)

    assert set(envelope) == {"content", "structured_content", "is_error"}
    assert envelope == {
        "content": [{"type": "text", "text": "sunny"}],
        "structured_content": {"temperature": 18},
        "is_error": False,
    }


def test_normalize_mcp_result_raises_structured_remote_error_for_is_error():
    result = CallToolResult(content=[TextContent(text="denied")], isError=True)

    with pytest.raises(RemoteToolError) as caught:
        normalize_mcp_result(result)

    assert caught.value.output == {
        "content": [{"type": "text", "text": "denied"}],
        "structured_content": None,
        "is_error": True,
    }


@pytest.mark.parametrize(
    "result",
    [
        type(
            "OpaqueContent",
            (),
            {"content": [object()], "structured_content": None, "is_error": False},
        )(),
        type(
            "OpaqueStructured",
            (),
            {"content": [], "structured_content": object(), "is_error": False},
        )(),
    ],
)
def test_normalize_mcp_result_rejects_non_json_data_with_a_safe_error(result):
    with pytest.raises(RemoteToolError) as caught:
        normalize_mcp_result(result)

    assert caught.value.output["is_error"] is True
    assert caught.value.output["content"] == []
    assert isinstance(caught.value.output["structured_content"]["error"], str)


@pytest.mark.parametrize(
    "structured",
    [
        {"value": math.nan},
        {1: "non-string key"},
        ("tuple",),
    ],
)
def test_normalize_mcp_result_rejects_values_strict_json_would_change(structured):
    result = CallToolResult(content=[], structuredContent=structured)

    with pytest.raises(RemoteToolError) as caught:
        normalize_mcp_result(result)

    assert caught.value.output["is_error"] is True
    assert "Invalid MCP result" in caught.value.output["structured_content"]["error"]


def test_make_mcp_tool_requires_approval_and_normalizes_the_invocation_result():
    class RemoteTool:
        name = "forecast"
        description = "Get a forecast."
        inputSchema = {"type": "object", "properties": {"city": {"type": "string"}}}

    calls = []

    async def invoke(name, arguments):
        calls.append((name, arguments))
        return CallToolResult(content=[TextContent(text="sunny")])

    tool = make_mcp_tool("weather", RemoteTool(), invoke)
    output = asyncio.run(tool.executer({"city": "Beijing"}))

    assert tool.id == tool.name == "mcp__weather__forecast"
    assert tool.origin is ToolOrigin.MCP
    assert tool.requires_approval is True
    assert tool.input_schema == RemoteTool.inputSchema
    assert calls == [("forecast", {"city": "Beijing"})]
    assert output["content"] == [{"type": "text", "text": "sunny"}]
