from src.core.context import build_tool_messages
from src.schema.event import EventType
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


def test_completed_tool_emits_the_tool_name():
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

    event = executor.event_handler.events[0]
    assert event.type is EventType.TOOL_EXECUTION_COMPLETED
    assert event.payload.name == "shell"
    assert event.payload.arguments == '{"command": "dir"}'
    assert event.payload.tool_call_id == "call_0_abc"
    assert event.task_id == "task-1"


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
