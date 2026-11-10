import pytest
from src.context.context_builder import DefaultContextBuilder
from src.core.guard import GuardError, HarnessGuard
from src.core.llm import LlmCallError
from src.core.loop import DEFAULT_LLM_ATTEMPT_DELAY_SECONDS, DEFAULT_MAX_LLM_ATTEMPTS, Harness
from src.core.task import Task
from src.core.tool import ToolExecutor
from src.schema.context import Context
from src.schema.event import EventType
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall
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


class RecordingOutput:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _harness(gateway, guard=None, event_handler=None) -> Harness:
    return Harness(
        "Be brief.",
        gateway,
        [],
        [],
        Executor(),
        DefaultContextBuilder(),
        harness_guard=guard or HarnessGuard(),
        event_handler=event_handler,
    )


def test_run_plans_a_new_task_and_returns_the_answer():
    gateway = Gateway()
    task = _task()
    harness = _harness(gateway)

    assert harness.run(task).output == "bought"

    assert gateway.planning_calls == [(False, TaskPlan)]
    assert task.state is TaskState.TASK_COMPLETED
    assert task.plan.goal == ["buy a phone"]
    assert task.plan.current == "compare"


def test_planning_run_emits_task_events():
    output = RecordingOutput()

    _harness(Gateway(), event_handler=output).run(_task())

    assert [event.type for event in output.events] == [
        EventType.TASK_STARTED,
        EventType.TASK_PLANNING,
        EventType.TASK_PLANNED,
        EventType.LLM_DELTA,
        EventType.TASK_COMPLETED,
    ]
    assert output.events[2].payload.goal == ["buy a phone"]


def test_planning_retry_emits_an_llm_retry_event(monkeypatch):
    monkeypatch.setattr("src.core.loop.time.sleep", lambda seconds: None)
    gateway = Gateway()
    gateway.failures = 1
    output = RecordingOutput()

    _harness(gateway, event_handler=output).run(_task())

    retries = [event for event in output.events if event.type is EventType.LLM_RETRYING]
    assert len(retries) == 1
    assert retries[0].payload.phase == "planning"
    assert retries[0].payload.attempt == 1
    assert retries[0].payload.reason == "TimeoutError"


def test_invalid_plan_json_is_retried(monkeypatch):
    monkeypatch.setattr("src.core.loop.time.sleep", lambda seconds: None)

    class InvalidJsonGateway(Gateway):
        def planning(self, context, stream=True, response_format=None):
            self.planning_calls.append((stream, response_format))
            if len(self.planning_calls) == 1:
                TaskPlan.model_validate_json("{")
            return LLMOutput(
                FinishReason.STOP,
                None,
                [],
                None,
                None,
                structured=TaskPlan(goal=["buy a phone"]),
            )

    output = RecordingOutput()
    task = _task()

    _harness(InvalidJsonGateway(), event_handler=output).run(task)

    assert task.state is TaskState.TASK_COMPLETED
    assert task.result.output == "bought"
    assert task.plan.goal == ["buy a phone"]
    retries = [event for event in output.events if event.type is EventType.LLM_RETRYING]
    assert len(retries) == 1
    assert retries[0].payload.phase == "planning"
    assert retries[0].payload.reason == "ValidationError"


def test_invalid_plan_json_fails_after_the_planning_attempts(monkeypatch):
    monkeypatch.setattr("src.core.loop.time.sleep", lambda seconds: None)

    class AlwaysInvalidGateway(Gateway):
        def planning(self, context, stream=True, response_format=None):
            self.planning_calls.append((stream, response_format))
            TaskPlan.model_validate_json("{")

    gateway = AlwaysInvalidGateway()

    with pytest.raises(LlmCallError, match="Invalid JSON"):
        _harness(gateway).run(_task())

    assert len(gateway.planning_calls) == DEFAULT_MAX_LLM_ATTEMPTS


def test_infer_streams_deltas_and_assembles_the_answer():
    class StreamGateway(Gateway):
        def infer(self, context, stream=True, response_format=None):
            assert stream is True
            yield LLMOutput(None, "hel", [], None, None)
            yield LLMOutput(FinishReason.STOP, "lo", [], None, None)

    output = RecordingOutput()
    harness = _harness(StreamGateway(), event_handler=output)
    task = _task()

    result = harness.infer(Context(messages=[], tools=[]), task)

    assert result.content == "hello"
    assert result.finish_reason is FinishReason.STOP
    deltas = [event for event in output.events if event.type is EventType.LLM_DELTA]
    assert [event.payload.content for event in deltas] == ["hel", "lo"]


def test_infer_hides_text_that_accompanies_tool_calls():
    class StreamGateway(Gateway):
        def infer(self, context, stream=True, response_format=None):
            yield LLMOutput(None, "checking", [], None, None)
            yield LLMOutput(
                FinishReason.TOOL_CALLS,
                None,
                [LlmToolCall("call_1", "shell", "{}")],
                None,
                None,
            )

    output = RecordingOutput()
    harness = _harness(StreamGateway(), event_handler=output)

    result = harness.infer(Context(messages=[], tools=[]), _task())

    assert result.content == "checking"
    assert result.tool_calls[0].name == "shell"
    assert [event for event in output.events if event.type is EventType.LLM_DELTA] == []


def test_planning_context_names_tools_but_does_not_offer_them():
    tool = Tool(
        id="shell",
        name="shell",
        description="Run a command.",
        input_schema={},
        executer=lambda inputs: "",
    )
    harness = Harness(
        "Be brief.",
        Gateway(),
        [tool],
        [],
        Executor(),
        DefaultContextBuilder(),
        harness_guard=HarnessGuard(),
    )

    context = harness.build_planning_context(_task())

    assert context.tools == []
    assert "shell: Run a command." in context.messages[0].content


def test_later_steps_keep_the_goal_and_drop_stale_plan_steps():
    task = _task()
    task.history = []
    task.plan = TaskPlan(goal=["look"], current="read", remaining=["summarize"])

    first = DefaultContextBuilder().build_infer_context("Be brief.", task, [], [])
    assert "summarize" in first.messages[-1].content

    task.history = [first.messages[1]]
    later = DefaultContextBuilder().build_infer_context("Be brief.", task, [], [])
    assert "look" in later.messages[-1].content
    assert "summarize" not in later.messages[-1].content


def test_transient_planning_errors_are_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.core.loop.time.sleep", lambda seconds: sleeps.append(seconds))
    gateway = Gateway()
    gateway.failures = 1

    assert _harness(gateway).run(_task()).output == "bought"

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
    assert task.result.output == ""
    assert _harness(Gateway()).run(task).output == ""


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
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
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
