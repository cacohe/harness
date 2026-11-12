import pytest
from src.core import tool as core_tool
from src.core.context import build_tool_messages
from src.core.tool import ToolExecutor
from src.schema.llm import LlmToolCall
from src.schema.tool import (
    ApprovalDecision,
    Tool,
    ToolExecutionFailReason,
    ToolExecutionState,
)
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry


def test_core_exposes_tool_execution_ports():
    assert hasattr(core_tool, "ToolCatalog")
    assert hasattr(core_tool, "ToolRunner")
    assert hasattr(core_tool, "ApprovalPolicy")
    assert hasattr(core_tool, "SandboxRunner")


class Runner:
    def __init__(self, output="ran", error=None):
        self.output = output
        self.error = error
        self.calls = []

    def run(self, tool, inputs):
        self.calls.append((tool, inputs))
        if self.error is not None:
            raise self.error
        return self.output


class Policy:
    def __init__(self, decision=ApprovalDecision.APPROVAL_DECISION_APPROVED):
        self.decision = decision

    def decide(self, tool, inputs):
        return self.decision


class Sandbox:
    def __init__(self, supported=True, output="sandboxed"):
        self.supported = supported
        self.output = output
        self.calls = []

    def supports(self, tool):
        return self.supported

    def run(self, tool, inputs):
        self.calls.append((tool, inputs))
        return self.output


def _core_executor(*, runner=None, policy=None, sandbox=None):
    registry = DefaultToolRegistry()
    registry.register(
        Tool(
            id="shell",
            name="shell",
            description="Run a command.",
            input_schema={},
            executer=lambda inputs: "legacy",
        )
    )
    return ToolExecutor(
        registry,
        runner if runner is not None else Runner(),
        approval_policy=policy if policy is not None else Policy(),
        sandbox_runner=sandbox,
    )


def test_unknown_tool_returns_not_supported():
    result = _core_executor().execute_tool_call(
        LlmToolCall(id="call-1", name="missing", arguments="{}")
    )

    assert result.state is ToolExecutionState.TOOL_EXECUTION_STATE_FAILED
    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_NOT_SUPPORTED


def test_non_object_arguments_fail_before_runner_is_called():
    runner = Runner()
    result = _core_executor(runner=runner).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments="[]")
    )

    assert result.state is ToolExecutionState.TOOL_EXECUTION_STATE_FAILED
    assert result.output == "tool arguments are not a JSON object"
    assert runner.calls == []


def test_runner_exception_becomes_a_failed_result():
    result = _core_executor(runner=Runner(error=RuntimeError("nope"))).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments="{}")
    )

    assert result.state is ToolExecutionState.TOOL_EXECUTION_STATE_FAILED
    assert result.output == "RuntimeError: nope"


def test_rejected_tool_does_not_run():
    runner = Runner()
    result = _core_executor(
        runner=runner, policy=Policy(ApprovalDecision.APPROVAL_DECISION_REJECTED)
    ).execute_tool_call(LlmToolCall(id="call-1", name="shell", arguments="{}"))

    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED
    assert runner.calls == []


def test_undecided_tool_is_pending_with_its_arguments():
    runner = Runner()
    result = _core_executor(runner=runner, policy=Policy(None)).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command": "dir"}')
    )

    assert result.state is ToolExecutionState.TOOL_EXECUTION_STATE_PENDING
    assert result.arguments == '{"command": "dir"}'
    assert runner.calls == []


def test_supported_tool_runs_only_inside_sandbox():
    runner = Runner()
    sandbox = Sandbox()
    result = _core_executor(runner=runner, sandbox=sandbox).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command": "dir"}')
    )

    assert result.state is ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED
    assert result.output == "sandboxed"
    assert len(sandbox.calls) == 1
    assert runner.calls == []


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
        [LlmToolCall(id="call_0_abc", name="read_file", arguments='{"path": "README.md"}')]
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
