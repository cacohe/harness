import asyncio
import threading
import time

import pytest
from src.core import tool as core_tool
from src.core.context import build_tool_messages
from src.core.tool import RemoteToolError, ToolExecutor
from src.core.workspace import Workspace, WorkspaceViolation
from src.schema.llm import LlmToolCall
from src.schema.tool import (
    ApprovalDecision,
    Tool,
    ToolExecutionFailReason,
    ToolExecutionState,
    ToolOrigin,
)
from src.tool.local_tools import LocalToolSuite
from src.tool.tool_executor import DefaultToolExecutor, LocalToolRunner
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


def test_local_io_failure_does_not_expose_the_host_workspace_path(tmp_path):
    workspace = Workspace(tmp_path)
    [tool] = [item for item in LocalToolSuite(workspace).tools() if item.name == "write_file"]
    executor = _executor_for_tool(tool, runner=LocalToolRunner())

    result = executor.execute_tool_call(
        LlmToolCall(id="call-1", name="write_file", arguments='{"path":".","content":"x"}'),
        approved=True,
    )

    assert result.is_failed()
    assert str(workspace.root) not in str(result.output).replace("\\\\", "\\")
    assert "workspace path ." in str(result.output)


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


def test_sandbox_timeout_is_classified_as_a_tool_timeout(tmp_path):
    class TimedOutSandbox(Sandbox):
        def run(self, tool, inputs, workspace):
            raise TimeoutError("sandbox command exceeded its timeout")

    tool = _sandbox_tool(Workspace(tmp_path))
    result = _executor_for_tool(tool, sandbox=TimedOutSandbox()).execute_tool_call(
        LlmToolCall(id="call-1", name="shell", arguments='{"command":"sleep 60"}')
    )

    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_TIMEOUT
    assert result.output == "TimeoutError: sandbox command exceeded its timeout"


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


def test_resumed_write_uses_normalized_target_after_symlink_alias_retarget(tmp_path):
    root = tmp_path / "workspace"
    inside = root / "inside"
    outside = tmp_path / "outside"
    root.mkdir()
    inside.mkdir()
    outside.mkdir()
    alias = root / "alias"
    try:
        alias.symlink_to(inside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    workspace = Workspace(root)
    [tool] = [item for item in LocalToolSuite(workspace).tools() if item.name == "write_file"]
    executor = _executor_for_tool(tool, runner=LocalToolRunner(), policy=Policy(None))
    held = executor.execute_tool_call(
        LlmToolCall(
            id="call-1",
            name="write_file",
            arguments='{"path":"alias/file.txt","content":"safe"}',
        )
    )
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED
    alias.unlink()
    alias.symlink_to(outside, target_is_directory=True)

    result = executor.execute_held_tool_call(held)

    assert result.is_completed()
    assert (inside / "file.txt").read_text(encoding="utf-8") == "safe"
    assert not (outside / "file.txt").exists()


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


def test_tool_call_result_keeps_original_mixed_call_order():
    executor = _executor(lambda inputs: "ran", "shell", "read_file")

    result = executor.execute_tool_calls(
        [
            LlmToolCall(id="call-1", name="shell", arguments='{"command":"status"}'),
            LlmToolCall(id="call-2", name="missing", arguments="{}"),
            LlmToolCall(id="call-3", name="read_file", arguments='{"path":"a"}'),
        ]
    )

    assert [item.tool_id for item in result.ordered_tools] == ["call-1", "call-2", "call-3"]


def test_cancellation_stops_later_tools_and_preserves_order():
    cancelled = False
    calls = []

    def run(inputs):
        nonlocal cancelled
        calls.append(inputs)
        cancelled = True
        return "ran"

    executor = _executor(run, "first", "second", "third")
    result = executor.execute_tool_calls(
        [
            LlmToolCall(id="call-1", name="first", arguments='{"value":1}'),
            LlmToolCall(id="call-2", name="second", arguments='{"value":2}'),
            LlmToolCall(id="call-3", name="third", arguments='{"value":3}'),
        ],
        should_cancel=lambda: cancelled,
    )

    assert calls == [{"value": 1}]
    assert [item.tool_id for item in result.ordered_tools] == ["call-1", "call-2", "call-3"]
    assert [item.tool_id for item in result.completed_tools] == ["call-1"]
    assert [item.tool_id for item in result.failed_tools] == ["call-2", "call-3"]
    assert all(
        item.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED
        and item.output == "Tool call cancelled."
        for item in result.failed_tools
    )


def test_cancelled_tools_skip_before_call_but_reach_after_call():
    executor = _executor(lambda inputs: "ran", "first", "second")
    before = []
    after = []

    result = executor.execute_tool_calls(
        [
            LlmToolCall(id="call-1", name="first", arguments="{}"),
            LlmToolCall(id="call-2", name="second", arguments="{}"),
        ],
        before_call=lambda call: before.append(call.id),
        after_call=lambda call, item: after.append((call.id, item.fail_reason)),
        should_cancel=lambda: True,
    )

    assert before == []
    assert [call_id for call_id, _ in after] == ["call-1", "call-2"]
    assert result.completed_tools == []


def test_precancelled_batch_never_invokes_policy_validator_or_runner():
    runner = Runner()
    policy = CapturingPolicy()
    validations = []

    def validate(inputs):
        validations.append(inputs)
        return inputs

    tool = Tool("read", "read", "Read.", {}, lambda inputs: "legacy", input_validator=validate)
    executor = _executor_for_tool(tool, runner=runner, policy=policy)

    result = executor.execute_tool_calls(
        [LlmToolCall(id="call-1", name="read", arguments='{"path":"a"}')],
        should_cancel=lambda: True,
    )

    assert validations == []
    assert policy.calls == []
    assert runner.calls == []
    assert result.failed_tools[0].fail_reason is (
        ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED
    )


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


def test_call_held_behind_another_honors_a_policy_rejection():
    runner = Runner()
    tool = Tool("read_file", "read_file", "Read.", {}, lambda inputs: "legacy")
    executor = _executor_for_tool(
        tool,
        runner=runner,
        policy=Policy(ApprovalDecision.APPROVAL_DECISION_REJECTED),
    )
    held = executor.execute_tool_call(
        LlmToolCall(id="call_2", name="read_file", arguments='{"path":"b"}'),
        hold=True,
    )

    result = executor.execute_held_tool_call(held)

    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED
    assert runner.calls == []


@pytest.mark.parametrize("name", ["read_file", "list_dir", "grep", "glob"])
def test_read_only_tools_run_without_approval(name):
    executor = _executor(lambda inputs: "ran", name)

    result = executor.execute_tool_calls([LlmToolCall(id="call_1", name=name, arguments="{}")])

    assert result.pending_tools == []
    assert result.completed_tools[0].output == "ran"


def test_async_executor_awaits_a_native_async_runner():
    class AsyncRunner:
        async def arun(self, tool, inputs):
            await asyncio.sleep(0)
            return {"tool": tool.name, "inputs": inputs}

        def run(self, tool, inputs):
            raise AssertionError("native async runner must be preferred")

    tool = Tool("read", "read", "Read.", {}, lambda inputs: "unused")
    executor = _executor_for_tool(tool, runner=AsyncRunner())

    result = asyncio.run(
        executor.aexecute_tool_call(LlmToolCall(id="call-1", name="read", arguments='{"path":"a"}'))
    )

    assert result.is_completed()
    assert result.output == {"tool": "read", "inputs": {"path": "a"}}


def test_async_executor_accepts_a_runner_that_implements_only_the_async_protocol():
    class AsyncOnlyRunner:
        async def arun(self, tool, inputs):
            return inputs["value"]

    tool = Tool("read", "read", "Read.", {}, lambda inputs: "unused")
    executor = _executor_for_tool(tool, runner=AsyncOnlyRunner())

    result = asyncio.run(
        executor.aexecute_tool_call(
            LlmToolCall(id="call-1", name="read", arguments='{"value":"native"}')
        )
    )

    assert result.output == "native"


def test_async_executor_classifies_a_native_async_runner_error():
    class AsyncRunner:
        async def arun(self, tool, inputs):
            raise RuntimeError("remote failed")

        def run(self, tool, inputs):
            raise AssertionError("native async runner must be preferred")

    tool = Tool("read", "read", "Read.", {}, lambda inputs: "unused")
    executor = _executor_for_tool(tool, runner=AsyncRunner())

    result = asyncio.run(
        executor.aexecute_tool_call(LlmToolCall(id="call-1", name="read", arguments="{}"))
    )

    assert result.is_failed()
    assert result.output == "RuntimeError: remote failed"


def test_async_executor_offloads_a_legacy_runner_without_blocking_the_loop():
    caller_thread = threading.get_ident()
    ticks: list[str] = []

    class BlockingRunner:
        def run(self, tool, inputs):
            time.sleep(0.05)
            return threading.get_ident()

    tool = Tool("read", "read", "Read.", {}, lambda inputs: "unused")
    executor = _executor_for_tool(tool, runner=BlockingRunner())

    async def scenario():
        call = asyncio.create_task(
            executor.aexecute_tool_call(LlmToolCall(id="call-1", name="read", arguments="{}"))
        )
        await asyncio.sleep(0.01)
        ticks.append("tick")
        return await call

    result = asyncio.run(scenario())

    assert ticks == ["tick"]
    assert result.output != caller_thread


def test_local_tool_runner_awaits_an_async_tool_callable():
    async def execute(inputs):
        await asyncio.sleep(0)
        return inputs["value"] * 2

    executor = _executor(execute, "double")

    result = asyncio.run(
        executor.aexecute_tool_call(
            LlmToolCall(id="call-1", name="double", arguments='{"value":3}')
        )
    )

    assert result.output == 6


def test_async_tool_batch_preserves_order_and_cancels_calls_after_the_boundary():
    cancelled = False
    calls: list[str] = []

    async def execute(inputs):
        nonlocal cancelled
        calls.append(inputs["name"])
        cancelled = True
        return inputs["name"]

    executor = _executor(execute, "first", "second")

    result = asyncio.run(
        executor.aexecute_tool_calls(
            [
                LlmToolCall("call-1", "first", '{"name":"first"}'),
                LlmToolCall("call-2", "second", '{"name":"second"}'),
            ],
            should_cancel=lambda: cancelled,
        )
    )

    assert calls == ["first"]
    assert [item.tool_id for item in result.ordered_tools] == ["call-1", "call-2"]
    assert result.failed_tools[0].fail_reason is (
        ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED
    )


def test_async_held_tool_call_resumes_after_approval():
    executor = _executor(lambda inputs: "ran", "shell")
    held = asyncio.run(
        executor.aexecute_tool_call(LlmToolCall("call-1", "shell", '{"command":"status"}'))
    )
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED

    result = asyncio.run(executor.aexecute_held_tool_call(held))

    assert result.is_completed()
    assert result.output == "ran"


def test_async_held_call_honors_a_legacy_execute_tool_call_override():
    class LegacyOverrideExecutor(DefaultToolExecutor):
        def __init__(self, registry):
            super().__init__(registry)
            self.legacy_calls = 0

        def execute_tool_call(self, call, approved=False, hold=False):
            self.legacy_calls += 1
            return super().execute_tool_call(call, approved=approved, hold=hold)

    registry = DefaultToolRegistry()
    registry.register(Tool("shell", "shell", "Run.", {}, lambda inputs: "ran"))
    executor = LegacyOverrideExecutor(registry)
    held = executor.execute_tool_call(LlmToolCall("call-1", "shell", "{}"))
    executor.legacy_calls = 0
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED

    result = asyncio.run(executor.aexecute_held_tool_call(held))

    assert result.output == "ran"
    assert executor.legacy_calls == 1


def test_sync_tool_wrapper_rejects_an_active_event_loop_before_execution():
    called = False

    def execute(inputs):
        nonlocal called
        called = True
        return "ran"

    executor = _executor(execute)

    async def scenario():
        with pytest.raises(RuntimeError, match="await aexecute_tool_call"):
            executor.execute_tool_call(LlmToolCall("call-1", "read_file", "{}"))

    asyncio.run(scenario())

    assert called is False


def test_local_tool_suite_marks_every_tool_as_local(tmp_path):
    tools = LocalToolSuite(Workspace(tmp_path)).tools(include_shell=True)

    assert tools
    assert {tool.origin for tool in tools} == {ToolOrigin.LOCAL}


def test_default_policy_pauses_a_tool_marked_as_requiring_approval():
    registry = DefaultToolRegistry()
    registry.register(
        Tool(
            "remote",
            "remote",
            "Remote.",
            {},
            lambda inputs: "ran",
            origin=ToolOrigin.MCP,
            requires_approval=True,
        )
    )
    executor = DefaultToolExecutor(registry)

    result = executor.execute_tool_call(LlmToolCall("call-1", "remote", "{}"))

    assert result.is_pending()
    assert result.needs_approval is True


def test_injected_policy_can_approve_a_tool_marked_as_requiring_approval():
    tool = Tool(
        "remote",
        "remote",
        "Remote.",
        {},
        lambda inputs: "unused",
        origin=ToolOrigin.MCP,
        requires_approval=True,
    )
    executor = _executor_for_tool(tool, runner=Runner("approved"), policy=Policy())

    result = executor.execute_tool_call(LlmToolCall("call-1", "remote", "{}"))

    assert result.is_completed()
    assert result.output == "approved"


def test_remote_tool_error_becomes_a_remote_failure_with_structured_output():
    output = {"content": [], "structured_content": None, "is_error": True}

    async def fail(_inputs):
        raise RemoteToolError(output)

    executor = _executor(fail, "remote")

    result = asyncio.run(executor.aexecute_tool_call(LlmToolCall("call-1", "remote", "{}")))

    assert result.is_failed()
    assert result.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REMOTE_ERROR
    assert result.output == output


def test_approved_remote_call_rejects_a_replacement_tool_generation():
    called = []
    registry = DefaultToolRegistry()
    registry.register(
        Tool(
            "remote",
            "remote",
            "Remote.",
            {},
            lambda inputs: "old",
            requires_approval=True,
            approval_scope="generation-old",
        )
    )
    executor = DefaultToolExecutor(registry)
    held = asyncio.run(executor.aexecute_tool_call(LlmToolCall("call-1", "remote", "{}")))
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED
    registry.unregister("remote")
    registry.register(
        Tool(
            "remote",
            "remote",
            "Replacement.",
            {},
            lambda inputs: called.append(inputs),
            requires_approval=True,
            approval_scope="generation-new",
        )
    )

    result = asyncio.run(executor.aexecute_held_tool_call(held))

    assert result.is_failed()
    assert result.fail_reason is (ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_APPROVAL_STALE)
    assert called == []


def test_legacy_super_override_keeps_native_async_tool_on_the_owner_loop():
    class LegacyExecutor(DefaultToolExecutor):
        def execute_tool_call(self, call, approved=False, hold=False):
            return super().execute_tool_call(call, approved=approved, hold=hold)

    async def scenario():
        loop = asyncio.get_running_loop()
        owner_future = loop.create_future()
        loop.call_later(0.01, owner_future.set_result, "owner-loop")

        async def execute(_inputs):
            return await owner_future

        registry = DefaultToolRegistry()
        registry.register(Tool("native", "native", "Native.", {}, execute))
        executor = LegacyExecutor(registry)
        result = await executor.aexecute_tool_calls([LlmToolCall("call-1", "native", "{}")])
        return result.ordered_tools[0]

    result = asyncio.run(scenario())

    assert result.is_completed()
    assert result.output == "owner-loop"
