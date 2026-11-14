import pytest
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


def test_register_many_strict_rejects_collisions_without_partial_changes():
    registry = DefaultToolRegistry()
    existing = Tool("existing", "existing", "Existing.", {}, lambda inputs: inputs)
    registry.register(existing)
    first = Tool("first", "first", "First.", {}, lambda inputs: inputs)
    collision = Tool("second", "existing", "Collision.", {}, lambda inputs: inputs)

    with pytest.raises(ValueError, match="collision"):
        registry.register_many_strict([first, collision])

    assert registry.get_all_tools() == [existing]


def test_register_many_strict_rejects_collisions_within_the_batch():
    registry = DefaultToolRegistry()
    first = Tool("first", "same", "First.", {}, lambda inputs: inputs)
    second = Tool("second", "same", "Second.", {}, lambda inputs: inputs)

    with pytest.raises(ValueError, match="collision"):
        registry.register_many_strict([first, second])

    assert registry.get_all_tools() == []


def test_register_many_strict_rejects_id_collisions_within_the_batch():
    registry = DefaultToolRegistry()
    first = Tool("same", "first", "First.", {}, lambda inputs: inputs)
    second = Tool("same", "second", "Second.", {}, lambda inputs: inputs)

    with pytest.raises(ValueError, match="collision"):
        registry.register_many_strict([first, second])

    assert registry.get_all_tools() == []
