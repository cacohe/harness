import pytest
from src.core import tool as core_tool
from src.core.context import build_tool_messages
from src.core.tool import ToolExecutor
from src.core.workspace import Workspace, WorkspaceViolation
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


class FalseyRunner(Runner):
    def __bool__(self):
        return False


class Policy:
    def __init__(self, decision=ApprovalDecision.APPROVAL_DECISION_APPROVED):
        self.decision = decision

    def decide(self, tool, inputs):
        return self.decision


class CapturingPolicy(Policy):
    def __init__(self, decision=ApprovalDecision.APPROVAL_DECISION_APPROVED):
        super().__init__(decision)
        self.calls = []

    def decide(self, tool, inputs):
        self.calls.append((tool, inputs))
        return self.decision


class FalseyPolicy(Policy):
    def __bool__(self):
        return False


class SequencePolicy:
    def __init__(self):
        self.calls = 0

    def decide(self, tool, inputs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("approval offline")
        return ApprovalDecision.APPROVAL_DECISION_APPROVED


class Sandbox:
    def __init__(self, supported=True, output="sandboxed"):
        self.supported = supported
        self.output = output
        self.calls = []

    def supports(self, tool):
        return self.supported

    def run(self, tool, inputs, workspace):
        self.calls.append((tool, inputs, workspace))
        return self.output


class FalseySandbox(Sandbox):
    def __bool__(self):
        return False


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


def _executor_for_tool(tool, *, runner=None, policy=None, sandbox=None):
    registry = DefaultToolRegistry()
    registry.register(tool)
    return ToolExecutor(
        registry,
        runner if runner is not None else Runner(),
        approval_policy=policy if policy is not None else Policy(),
        sandbox_runner=sandbox,
    )


def _sandbox_tool(workspace):
    return Tool(
        id="shell",
        name="shell",
        description="Run a command.",
        input_schema={},
        executer=lambda inputs: "host fallback",
        requires_sandbox=True,
        input_validator=lambda inputs: {**inputs, "cwd": inputs.get("cwd") or "."},
        workspace=workspace,
    )


def test_validator_output_is_used_for_approval_and_execution():
    runner = Runner()
    policy = CapturingPolicy()
    tool = Tool(
        id="read_file",
        name="read_file",
        description="Read a file.",
        input_schema={},
        executer=lambda inputs: "legacy",
        input_validator=lambda inputs: {**inputs, "path": "normalized.txt"},
    )
    executor = _executor_for_tool(tool, runner=runner, policy=policy)

    result = executor.execute_tool_call(
        LlmToolCall(id="call-1", name="read_file", arguments='{"path":"alias.txt"}')
    )

    assert result.is_completed()
    assert policy.calls == [(tool, {"path": "normalized.txt"})]
    assert runner.calls == [(tool, {"path": "normalized.txt"})]


def test_workspace_violation_fails_before_policy_or_runner():
    runner = Runner()
    policy = CapturingPolicy()

    def reject(_inputs):
        raise WorkspaceViolation("blocked")

    tool = Tool(
        id="read_file",
        name="read_file",
        description="Read a file.",
        input_schema={},
        executer=lambda inputs: "legacy",
        input_validator=reject,
    )
    executor = _executor_for_tool(tool, runner=runner, policy=policy)

    result = executor.execute_tool_call(
        LlmToolCall(id="call-1", name="read_file", arguments='{"path":"../outside"}')
    )

    assert (
        result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION
    )
    assert result.output == "WorkspaceViolation: blocked"
    assert policy.calls == []
    assert runner.calls == []


def test_held_call_persists_normalized_arguments_and_workspace_identity(tmp_path):
    workspace = Workspace(tmp_path)
    tool = Tool(
        id="write_file",
        name="write_file",
        description="Write a file.",
        input_schema={},
        executer=lambda inputs: "legacy",
        input_validator=lambda inputs: {**inputs, "path": "nested/file.txt"},
        workspace=workspace,
    )
    executor = _executor_for_tool(tool, policy=Policy(None))

    result = executor.execute_tool_call(
        LlmToolCall(
            id="call-1",
            name="write_file",
            arguments='{"path": "alias.txt", "content": "hello"}',
        )
    )

    assert result.is_pending()
    assert result.arguments == '{"content":"hello","path":"nested/file.txt"}'
    assert result.workspace_id == workspace.identity


def test_resumed_call_revalidates_normalized_arguments(tmp_path):
    workspace = Workspace(tmp_path)
    validations = []
    runner = Runner()

    def validate(inputs):
        validations.append(dict(inputs))
        if len(validations) == 2:
            raise WorkspaceViolation("policy changed")
        return dict(inputs)

    tool = Tool(
        id="write_file",
        name="write_file",
        description="Write a file.",
        input_schema={},
        executer=lambda inputs: "legacy",
        input_validator=validate,
        workspace=workspace,
    )
    executor = _executor_for_tool(tool, runner=runner, policy=Policy(None))
    held = executor.execute_tool_call(
        LlmToolCall(id="call-1", name="write_file", arguments='{"path":"file.txt"}')
    )
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED

    result = executor.execute_held_tool_call(held)

    assert (
        result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION
    )
    assert validations == [{"path": "file.txt"}, {"path": "file.txt"}]
    assert runner.calls == []


def test_sandbox_required_tool_fails_without_sandbox_and_never_uses_runner(tmp_path):
    runner = Runner()
    tool = _sandbox_tool(Workspace(tmp_path))

    result = _executor_for_tool(tool, runner=runner).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )

    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_SANDBOX_REQUIRED
    assert runner.calls == []


def test_sandbox_required_tool_fails_when_sandbox_does_not_support_it(tmp_path):
    runner = Runner()
    sandbox = Sandbox(supported=False)
    tool = _sandbox_tool(Workspace(tmp_path))

    result = _executor_for_tool(tool, runner=runner, sandbox=sandbox).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )

    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_SANDBOX_REQUIRED
    assert sandbox.calls == []
    assert runner.calls == []


def test_sandbox_capability_error_fails_closed(tmp_path):
    class BrokenSandbox(Sandbox):
        def supports(self, tool):
            raise RuntimeError("sandbox offline")

    runner = Runner()
    tool = _sandbox_tool(Workspace(tmp_path))

    result = _executor_for_tool(tool, runner=runner, sandbox=BrokenSandbox()).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )

    assert result.is_failed()
    assert result.output == "RuntimeError: sandbox offline"
    assert runner.calls == []


def test_sandbox_required_tool_without_workspace_fails_before_sandbox():
    runner = Runner()
    sandbox = Sandbox()
    tool = _sandbox_tool(None)

    result = _executor_for_tool(tool, runner=runner, sandbox=sandbox).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )

    assert (
        result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION
    )
    assert result.output == "WorkspaceViolation: sandbox-required tool has no bound workspace"
    assert sandbox.calls == []
    assert runner.calls == []


def test_sandbox_receives_tool_inputs_and_bound_workspace(tmp_path):
    workspace = Workspace(tmp_path)
    runner = Runner()
    sandbox = Sandbox()
    tool = _sandbox_tool(workspace)

    result = _executor_for_tool(tool, runner=runner, sandbox=sandbox).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )

    assert result.output == "sandboxed"
    assert sandbox.calls == [(tool, {"command": "dir", "cwd": "."}, workspace)]
    assert runner.calls == []


def test_regular_tool_uses_bound_runner_even_when_sandbox_supports_it():
    runner = Runner()
    sandbox = Sandbox()
    tool = Tool("read_file", "read_file", "Read.", {}, lambda inputs: "legacy")

    result = _executor_for_tool(tool, runner=runner, sandbox=sandbox).execute_tool_call(
        LlmToolCall(id="call-1", name="read_file", arguments='{"path":"a.txt"}')
    )

    assert result.output == "ran"
    assert runner.calls == [(tool, {"path": "a.txt"})]
    assert sandbox.calls == []


def test_falsey_sandbox_is_preserved(tmp_path):
    workspace = Workspace(tmp_path)
    sandbox = FalseySandbox()
    tool = _sandbox_tool(workspace)

    result = _executor_for_tool(tool, sandbox=sandbox).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )

    assert result.output == "sandboxed"
    assert sandbox.calls == [(tool, {"command": "dir", "cwd": "."}, workspace)]


def test_resume_rejects_approval_from_another_workspace(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_tool = _sandbox_tool(Workspace(first_root))
    second_tool = _sandbox_tool(Workspace(second_root))
    executor = _executor_for_tool(first_tool, policy=Policy(None), sandbox=Sandbox())
    held = executor.execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"dir"}')
    )
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED
    executor.catalog.register(second_tool)

    result = executor.execute_held_tool_call(held)

    assert (
        result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION
    )
    assert result.output == "WorkspaceViolation: approval belongs to a different workspace"


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
    assert result.arguments == '{"command":"dir"}'
    assert runner.calls == []


def test_falsey_rejecting_policy_is_not_replaced_by_allow_all():
    runner = Runner()
    policy = FalseyPolicy(ApprovalDecision.APPROVAL_DECISION_REJECTED)

    result = _core_executor(runner=runner, policy=policy).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments="{}")
    )

    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED
    assert runner.calls == []


def test_falsey_custom_runner_is_not_replaced_by_local_runner():
    runner = FalseyRunner(output="custom")
    registry = DefaultToolRegistry()
    registry.register(
        Tool(
            id="read_file",
            name="read_file",
            description="Read a file.",
            input_schema={},
            executer=lambda inputs: "legacy",
        )
    )
    executor = DefaultToolExecutor(registry, runner=runner)

    result = executor.execute_tool_call(LlmToolCall(id="call-1", name="read_file", arguments="{}"))

    assert result.output == "custom"
    assert len(runner.calls) == 1


def test_approval_exception_fails_only_that_call_and_batch_continues():
    runner = Runner()
    executor = _core_executor(runner=runner, policy=SequencePolicy())

    result = executor.execute_tool_calls(
        [
            LlmToolCall(id="call-1", name="shell", arguments="{}"),
            LlmToolCall(id="call-2", name="shell", arguments="{}"),
        ]
    )

    assert [item.tool_id for item in result.completed_tools] == ["call-1"]
    assert [item.tool_id for item in result.failed_tools] == ["call-2"]
    assert result.failed_tools[0].output == "RuntimeError: approval offline"
    assert len(runner.calls) == 1


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
    assert result.pending_tools[0].arguments == '{"command":"git status"}'


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
