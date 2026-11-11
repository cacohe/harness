from src.core.context import build_tool_messages
from src.schema.llm import LlmToolCall
from src.schema.tool import Tool
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry


def _executor(executer) -> DefaultToolExecutor:
    registry = DefaultToolRegistry()
    registry.tools = {}
    registry.register(
        Tool(
            id="shell",
            name="shell",
            description="Run a command.",
            input_schema={},
            executer=executer,
        )
    )
    return DefaultToolExecutor(registry)


def test_completed_tool_message_keeps_the_model_call_id():
    executor = _executor(lambda inputs: "listed")

    result = executor.execute_tool_calls(
        [LlmToolCall(id="call_0_abc", name="shell", arguments='{"command": "dir"}')]
    )

    assert result.completed_tools[0].tool_name == "shell"
    assert result.completed_tools[0].output == "listed"
    message = build_tool_messages(result)[0]
    assert message.tool_call_id == "call_0_abc"
    assert message.name == "shell"


def test_executor_does_not_emit_events():
    class Handler:
        def __init__(self):
            self.events = []

        def emit(self, event, channel=None):
            self.events.append(event)

    executor = _executor(lambda inputs: "listed")
    executor.event_handler = Handler()

    executor.execute_tool_calls(
        [LlmToolCall(id="call_0_abc", name="shell", arguments='{"command": "dir"}')],
        task_id="task-1",
    )

    assert executor.event_handler.events == []


def test_failed_tool_message_keeps_the_model_call_id():
    def boom(inputs):
        raise RuntimeError("nope")

    executor = _executor(boom)

    result = executor.execute_tool_calls(
        [LlmToolCall(id="call_0_abc", name="shell", arguments="{}")]
    )

    message = build_tool_messages(result)[0]
    assert message.tool_call_id == "call_0_abc"
    assert message.name == "shell"
    assert message.content == "RuntimeError: nope"
