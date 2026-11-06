import pytest
from src.context.context_builder import DefaultContextBuilder
from src.core.guard import GuardError, HarnessGuard
from src.core.llm import LlmCallError
from src.core.loop import DEFAULT_LLM_ATTEMPT_DELAY_SECONDS, Harness
from src.core.task import Task
from src.core.tool import ToolExecutor
from src.schema.context import Context
from src.schema.harness import HarnessState
from src.schema.llm import FinishReason, LLMOutput
from src.schema.task import (
    TaskFailReason,
    TaskPauseReason,
    TaskPlan,
    TaskResult,
    TaskRunningReason,
    TaskState,
)
from src.schema.tool import Tool, ToolExecutionResult, ToolExecutionState


class Gateway:
    def __init__(self):
        self.planning_calls = []
        self.failures = 0

    def planning(self, context, stream=True, response_format=None):
        self.planning_calls.append((stream, response_format))
        if self.failures:
            self.failures -= 1
            raise TimeoutError("timed out")
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlan(goal=["buy a phone"], current="compare", remaining=["pay"]),
        )

    def infer(self, context, stream=True, response_format=None):
        return LLMOutput(FinishReason.STOP, "bought", [], None, None)


class Executor(ToolExecutor):
    def is_need_approval(self, tool: Tool) -> bool:
        return False

    def launch_approval(self, tool: Tool) -> bool:
        return False

    def is_need_sandbox(self, tool: Tool) -> bool:
        return False

    def get_sandbox(self):
        raise NotImplementedError

    def find_tool(self, name: str) -> Tool | None:
        return None

    def execute_without_sandbox(self, tool_id: str, inputs: dict) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_id=tool_id,
            tool_name=tool_id,
            state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
            output=None,
        )

    def execute(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        return self.execute_without_sandbox(tool.id, inputs)


def _task() -> Task:
    return Task(original_input="buy a phone")


def _harness(gateway, guard=None) -> Harness:
    return Harness(
        "Be brief.",
        gateway,
        [],
        [],
        Executor(),
        DefaultContextBuilder(),
        harness_guard=guard or HarnessGuard(),
    )


def test_run_plans_a_new_task_and_leaves_it_running():
    gateway = Gateway()
    task = _task()
    harness = _harness(gateway)

    assert harness.run(task).output == ""

    assert gateway.planning_calls == [(False, TaskPlan)]
    assert task.state is TaskState.TASK_RUNNING
    assert task.state_reason is TaskRunningReason.TASK_RUNNING_CREATE
    assert task.plan.goal == ["buy a phone"]
    assert task.plan.current == "compare"
    assert harness.state is HarnessState.HARNESS_STATE_PLANNING


def test_transient_planning_errors_are_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.core.loop.time.sleep", lambda seconds: sleeps.append(seconds))
    gateway = Gateway()
    gateway.failures = 1

    assert _harness(gateway).run(_task()).output == ""

    assert sleeps == [DEFAULT_LLM_ATTEMPT_DELAY_SECONDS]
    assert len(gateway.planning_calls) == 2


def test_permanent_planning_errors_fail_once():
    class FailingGateway(Gateway):
        def planning(self, context, stream=True, response_format=None):
            self.planning_calls.append((stream, response_format))
            raise ValueError("bad request")

    gateway = FailingGateway()
    task = _task()

    with pytest.raises(LlmCallError, match="bad request"):
        _harness(gateway).run(task)

    assert len(gateway.planning_calls) == 1
    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_LLM_ERROR


def test_guard_failure_keeps_its_own_reason():
    class Reject(HarnessGuard):
        def guard_input(self, task):
            raise GuardError("override the harness")

    task = _task()

    with pytest.raises(GuardError):
        _harness(Gateway(), guard=Reject()).run(task)

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_GUARD_ERROR


def test_context_builder_failure_is_not_replaced():
    class BrokenBuilder(DefaultContextBuilder):
        def build_planning_context(self, system_prompt, task, **kwargs):
            raise RuntimeError("missing memory")

    task = _task()
    harness = Harness(
        "Be brief.",
        Gateway(),
        [],
        [],
        Executor(),
        BrokenBuilder(),
        harness_guard=HarnessGuard(),
    )

    with pytest.raises(RuntimeError, match="missing memory"):
        harness.build_planning_context(task)

    assert task.state_reason is TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR


def test_paused_task_returns_to_running_without_planning():
    gateway = Gateway()
    task = _task()
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    task.transition_to_paused(TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL)
    task.plan = TaskPlan(goal=["buy a phone"])

    assert _harness(gateway).run(task).output == "bought"

    assert gateway.planning_calls == []
    assert task.state is TaskState.TASK_COMPLETED
    assert task.result.output == "bought"


def test_finished_task_returns_its_result():
    task = _task()
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    task.set_result(TaskResult("bought"))
    task.transition_to_completed()

    assert _harness(Gateway()).run(task).output == "bought"
    assert task.state is TaskState.TASK_COMPLETED


def test_builder_reads_history_from_the_task():
    task = _task()
    task.history = []
    task.plan = TaskPlan(goal=[])
    context = DefaultContextBuilder().build_infer_context("Be brief.", task, [], [])

    assert isinstance(context, Context)
    assert context.messages[0].role == "system"
    assert context.messages[1].content == "buy a phone"
    assert context.tools == []
