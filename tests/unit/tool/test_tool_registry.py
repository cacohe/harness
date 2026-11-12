from src.schema.tool import Tool
from src.tool.tool_registry import DefaultToolRegistry


def test_registries_do_not_share_tools():
    first = DefaultToolRegistry()
    second = DefaultToolRegistry()
    tool = Tool(
        id="one",
        name="one",
        description="One tool",
        input_schema={},
        executer=lambda inputs: inputs,
    )

    first.register(tool)

    assert first.get_all_tools() == [tool]
    assert second.get_all_tools() == []
