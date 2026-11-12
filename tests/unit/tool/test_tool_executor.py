import pytest
from src.core.context import build_tool_messages
from src.schema.llm import LlmToolCall
from src.schema.tool import (
    ApprovalDecision,
    Tool,
    ToolExecutionFailReason,
    ToolExecutionState,
)
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry


def _executor(executer, *names: str) -> DefaultToolExecutor:
    registry = DefaultToolRegistry()
    registry.tools = {}
    for name in names or ("read_file",):
        registry.register(
            Tool(
                id=name,
                name=name,
                description="A tool.",
                input_schema={},
                executer=executer,
            )
        )
    return DefaultToolExecutor(registry)


def test_completed_tool_message_keeps_the_model_call_id():
    executor = _executor(lambda inputs: "listed")

    result = executor.execute_tool_calls(
        [LlmToolCall(id="call_0_abc", name="read_file", arguments='{"path": "README.md"}')]
    )

    assert result.completed_tools[0].tool_name == "read_file"
    assert result.completed_tools[0].output == "listed"
    message = build_tool_messages(result)[0]
    assert message.tool_call_id == "call_0_abc"
    assert message.name == "read_file"


def test_executor_does_not_emit_events():
    class Handler:
        def __init__(self):
            self.events = []

        def emit(self, event, channel=None):
            self.events.append(event)

    executor = _executor(lambda inputs: "listed")
    executor.event_handler = Handler()

    executor.execute_tool_calls(
        [LlmToolCall(id="call_0_abc", name="read_file", arguments='{"path": "README.md"}')],
        task_id="task-1",
    )

    assert executor.event_handler.events == []


def test_failed_tool_message_keeps_the_model_call_id():
    def boom(inputs):
        raise RuntimeError("nope")

    executor = _executor(boom)

    result = executor.execute_tool_calls(
        [LlmToolCall(id="call_0_abc", name="read_file", arguments="{}")]
    )

    message = build_tool_messages(result)[0]
    assert message.tool_call_id == "call_0_abc"
    assert message.name == "read_file"
    assert message.content == "RuntimeError: nope"


@pytest.mark.parametrize("name", ["shell", "write_file", "edit_file"])
def test_mutating_tools_wait_for_approval_without_running(name):
    called = []

    def record(inputs):
        called.append(inputs)
        return "ran"

    executor = _executor(record, name)

    result = executor.execute_tool_calls(
        [LlmToolCall(id="call_1", name=name, arguments='{"command": "git status"}')]
    )

    assert called == []
    assert result.completed_tools == []
    assert result.pending_tools[0].state is ToolExecutionState.TOOL_EXECUTION_STATE_PENDING
    assert result.pending_tools[0].tool_id == "call_1"
    assert result.pending_tools[0].arguments == '{"command": "git status"}'


def test_calls_after_a_held_call_wait_in_order():
    called = []
    executor = _executor(lambda inputs: called.append(inputs) or "ran", "shell", "read_file")

    result = executor.execute_tool_calls(
        [
            LlmToolCall(id="call_1", name="read_file", arguments='{"path": "a"}'),
            LlmToolCall(id="call_2", name="shell", arguments='{"command": "git status"}'),
            LlmToolCall(id="call_3", name="read_file", arguments='{"path": "b"}'),
        ]
    )

    assert called == [{"path": "a"}]
    assert [item.tool_id for item in result.pending_tools] == ["call_2", "call_3"]
    assert [item.needs_approval for item in result.pending_tools] == [True, False]


def test_approved_call_runs_the_held_tool():
    called = []
    executor = _executor(lambda inputs: called.append(inputs) or "ran", "shell")
    call = LlmToolCall(id="call_1", name="shell", arguments='{"command": "git status"}')
    held = executor.execute_tool_call(call)
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED

    result = executor.execute_held_tool_call(held)

    assert called == [{"command": "git status"}]
    assert result.is_completed()
    assert result.tool_id == "call_1"


def test_rejected_call_fails_without_running():
    called = []
    executor = _executor(lambda inputs: called.append(inputs) or "ran", "shell")
    call = LlmToolCall(id="call_1", name="shell", arguments='{"command": "git status"}')
    held = executor.execute_tool_call(call)
    held.approval = ApprovalDecision.APPROVAL_DECISION_REJECTED

    result = executor.execute_held_tool_call(held)

    assert called == []
    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED
    assert result.output == "The tool call was rejected."


def test_call_held_behind_another_runs_without_a_decision():
    called = []
    executor = _executor(lambda inputs: called.append(inputs) or "ran")
    call = LlmToolCall(id="call_2", name="read_file", arguments='{"path": "b"}')
    held = executor.execute_tool_call(call, hold=True)

    assert held.is_pending()
    assert called == []

    result = executor.execute_held_tool_call(held)

    assert result.is_completed()
    assert called == [{"path": "b"}]


@pytest.mark.parametrize("name", ["read_file", "list_dir", "grep", "glob"])
def test_read_only_tools_run_without_approval(name):
    executor = _executor(lambda inputs: "ran", name)

    result = executor.execute_tool_calls([LlmToolCall(id="call_1", name=name, arguments="{}")])

    assert result.pending_tools == []
    assert result.completed_tools[0].output == "ran"
