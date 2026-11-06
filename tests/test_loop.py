import pytest
from src.context.context_builder import DefaultContextBuilder
from src.core.context import Context
from src.core.guard import GuardError
from src.core.hook import HarnessHook
from src.core.llm import FinishReason, LlmCallError, LLMOutput, LlmToolCall
from src.core.loop import DEFAULT_LLM_ATTEMPT_DELAY_SECONDS, Harness
from src.core.task import Task, TaskFailReason, TaskPauseReason, TaskState
from src.core.tool import ToolResult
from src.llm.plan import TaskPlanSchema


class Gateway:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.planning_calls = []
        self.contexts = []
        self.failures = 0

    def planning(self, context, stream=True, response_format=None):
        self.planning_calls.append((stream, response_format))
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlanSchema(goal=["buy a phone"], current="compare", remaining=["pay"]),
        )

    def infer(self, context, stream=True, response_format=None):
        assert stream is False
        self.contexts.append(context)
        if self.failures:
            self.failures -= 1
            raise TimeoutError("timed out")
        return self.outputs.pop(0)


class Executor:
    def __init__(self, approval=None):
        self.approval = set(approval or [])
        self.calls = []

    def is_need_approval(self, tool_name):
        return tool_name in self.approval

    def execute(self, tool_name, inputs):
        self.calls.append((tool_name, inputs))
        if tool_name == "read_file":
            return ToolResult(is_success=False, content=None, error="disk full")
        return f"ran {tool_name}"


def _task() -> Task:
    return Task(original_input="buy a phone", tools=[], skills=[])


def _output(reason, content="", calls=None, reasoning=None):
    return LLMOutput(reason, content, calls or [], reasoning, None)


def _call(call_id, name, arguments="{}"):
    return LlmToolCall(call_id, name, arguments)


def _harness(gateway, executor, max_steps=10, hooks=None):
    return Harness(
        gateway,
        executor,
        DefaultContextBuilder(),
        max_steps=max_steps,
        hooks=hooks,
    )


def test_run_plans_once_then_executes_every_tool_call():
    gateway = Gateway(
        [
            _output(
                FinishReason.TOOL_CALLS,
                calls=[_call("call_a", "read_file", '{"path": "a.txt"}'), _call("call_b", "shell")],
                reasoning="compare prices",
            ),
            _output(FinishReason.STOP, "bought"),
        ]
    )
    executor = Executor()
    seen = []
    harness = _harness(
        gateway,
        executor,
        hooks=HarnessHook(
            before_think=lambda task, context: seen.append("before"),
            after_think=lambda task, output: seen.append(output.finish_reason),
            before_execute_tool_calls=lambda task, output: seen.append("tools"),
            after_execute_tool_calls=lambda task, messages: seen.append(len(messages)),
        ),
    )

    assert harness.run(_task()) == "bought"

    assert gateway.planning_calls == [(False, TaskPlanSchema)]
    assert executor.calls == [("read_file", {"path": "a.txt"}), ("shell", {})]
    transcript = gateway.contexts[1].messages
    assert [item.role for item in transcript] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert transcript[1].content == "buy a phone"
    assert "Goal: ['buy a phone']" in transcript[-1].content
    assert transcript[2].reasoning_content == "compare prices"
    assert [item.content for item in transcript[3:5]] == ["disk full", "ran shell"]
    assert seen == ["before", FinishReason.TOOL_CALLS, "tools", 2, "before", FinishReason.STOP]


def test_run_records_the_completed_task(monkeypatch):
    gateway = Gateway([_output(FinishReason.STOP, "bought")])
    task = _task()

    assert Harness(gateway, Executor(), DefaultContextBuilder()).run(task) == "bought"

    assert task.state is TaskState.TASK_COMPLETED
    assert task.result == "bought"
    assert task.plan.goal == ["buy a phone"]
    assert task.plan.current == "compare"


def test_tool_argument_errors_go_back_to_the_model():
    gateway = Gateway(
        [
            _output(FinishReason.TOOL_CALLS, calls=[_call("call_a", "shell", "{")]),
            _output(FinishReason.STOP, "fixed"),
        ]
    )
    executor = Executor()
    task = _task()

    assert Harness(gateway, executor, DefaultContextBuilder()).run(task) == "fixed"

    assert executor.calls == []
    assert gateway.contexts[1].messages[3].content.startswith("Error:")
    assert task.state is TaskState.TASK_COMPLETED


def test_truncated_output_fails_the_task():
    gateway = Gateway([_output(FinishReason.LENGTH, "partial")])
    executor = Executor()
    task = _task()

    assert Harness(gateway, executor, DefaultContextBuilder()).run(task) == ""

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_LLM_ERROR
    assert executor.calls == []


def test_max_steps_stops_before_another_model_call():
    gateway = Gateway([_output(FinishReason.TOOL_CALLS, calls=[_call("call_a", "shell")])] * 3)
    task = _task()

    assert _harness(gateway, Executor(), max_steps=1).run(task) == ""

    assert task.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS
    assert len(gateway.contexts) == 1


def test_approval_pauses_and_resume_continues_the_pending_call():
    gateway = Gateway(
        [
            _output(
                FinishReason.TOOL_CALLS,
                calls=[_call("call_a", "read_file"), _call("call_b", "shell")],
            ),
            _output(FinishReason.STOP, "done"),
        ]
    )
    executor = Executor(approval={"shell"})
    harness = _harness(gateway, executor)
    task = _task()

    assert harness.run(task) == ""

    assert task.state is TaskState.TASK_PAUSED
    assert task.state_reason is TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL
    assert [call.name for call in task.pending_tool_calls] == ["shell"]
    assert executor.calls == [("read_file", {})]
    assert [item.role for item in task.history] == ["assistant", "tool"]

    executor.approval.clear()
    assert harness.run(task) == "done"
    assert task.state is TaskState.TASK_COMPLETED
    assert [call[0] for call in executor.calls] == ["read_file", "shell"]


def test_transient_model_errors_are_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.core.loop.time.sleep", lambda seconds: sleeps.append(seconds))
    gateway = Gateway([_output(FinishReason.STOP, "bought")])
    gateway.failures = 1

    assert Harness(gateway, Executor(), DefaultContextBuilder()).run(_task()) == "bought"

    assert sleeps == [DEFAULT_LLM_ATTEMPT_DELAY_SECONDS]


def test_permanent_model_errors_fail_once():
    class FailingGateway(Gateway):
        def infer(self, context, stream=True, response_format=None):
            self.contexts.append(context)
            raise ValueError("bad request")

    gateway = FailingGateway([])
    task = _task()

    with pytest.raises(LlmCallError, match="bad request"):
        Harness(gateway, Executor(), DefaultContextBuilder()).run(task)

    assert len(gateway.contexts) == 1
    assert task.state_reason is TaskFailReason.TASK_FAILED_LLM_ERROR


def test_guard_failure_keeps_its_own_reason(monkeypatch):
    def reject(task):
        raise GuardError("override the harness")

    monkeypatch.setattr("src.core.loop.guard_input", reject)
    task = _task()

    with pytest.raises(GuardError):
        Harness(Gateway([]), Executor(), DefaultContextBuilder()).run(task)

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_GUARD_ERROR


def test_context_builder_failure_is_not_replaced():
    class BrokenBuilder(DefaultContextBuilder):
        def build_planning_context(self, system_prompt, task, **kwargs):
            raise RuntimeError("missing memory")

    task = _task()

    with pytest.raises(RuntimeError, match="missing memory"):
        Harness(Gateway([]), Executor(), BrokenBuilder()).run(task)

    assert task.state_reason is TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR


def test_builder_accepts_the_history_argument():
    task = _task()
    context = DefaultContextBuilder().build_infer_context("Be brief.", task, history=[])

    assert isinstance(context, Context)
    assert context.messages[0].role == "system"
    assert context.tools == []
