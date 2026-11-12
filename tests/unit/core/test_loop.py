import pytest
from src.context.context_builder import DefaultContextBuilder
from src.core.context import HELD_TOOL_MESSAGE, PENDING_TOOL_MESSAGE
from src.core.guard import NoOpGuard
from src.core.hook import HarnessHook
from src.core.loop import (
    DEFAULT_LLM_ATTEMPT_DELAY_SECONDS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_STEPS,
    Harness,
)
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
from src.schema.tool import (
    Tool,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry


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
    def __init__(self, tools=None):
        registry = DefaultToolRegistry()
        for tool in tools or []:
            registry.register(tool)
        super().__init__(registry, LocalRunner())


class LocalRunner:
    def run(self, tool, inputs):
        return tool.executer(inputs)


def _task() -> Task:
    task = Task(original_input="buy a phone")
    task.history = []
    task.pending_tool_calls = []
    return task


class RecordingOutput:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_harness_default_collaborators_are_isolated():
    first = Harness("", None, [], Executor(), None)
    second = Harness("", None, [], Executor(), None)

    assert first.harness_hooks is not second.harness_hooks
    assert first.harness_guard is not second.harness_guard


def _harness(
    gateway,
    guard=None,
    event_handler=None,
    max_steps=DEFAULT_MAX_STEPS,
    max_seconds=DEFAULT_MAX_SECONDS,
    hooks=None,
    tools=None,
    executor=None,
    repository=None,
) -> Harness:
    return Harness(
        "Be brief.",
        gateway,
        [],
        executor or Executor(tools),
        DefaultContextBuilder(),
        harness_hooks=hooks or HarnessHook(),
        harness_guard=guard or NoOpGuard(),
        event_handler=event_handler,
        task_repository=repository,
        max_steps=max_steps,
        max_seconds=max_seconds,
    )


class RecordingRepository:
    def __init__(self):
        self.tasks = {}
        self.states = []

    def save(self, task):
        self.tasks[task.id] = task
        self.states.append(task.state)

    def get(self, task_id):
        return self.tasks.get(task_id)


class ScriptedGateway(Gateway):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.infer_contexts = []

    def infer(self, context, stream=True, response_format=None):
        self.infer_contexts.append(context)
        if not self.answers:
            raise AssertionError("unexpected infer")
        return self.answers.pop(0)


class CountingExecutor(Executor):
    def __init__(self, tools=None):
        super().__init__(tools)
        self.batches = 0

    def execute_tool_call(self, call, approved=False, hold=False):
        self.batches += 1
        return super().execute_tool_call(call)


def _shell() -> Tool:
    return Tool(
        id="shell",
        name="shell",
        description="Run a command.",
        input_schema={},
        executer=lambda inputs: "",
    )


def _model(reason, content=None, calls=None) -> LLMOutput:
    return LLMOutput(reason, content, calls or [], None, None)


def test_run_plans_a_new_task_and_returns_the_answer():
    gateway = Gateway()
    task = _task()
    harness = _harness(gateway)

    assert harness.run(task).output == "bought"

    assert gateway.planning_calls == [(False, TaskPlan)]


def test_run_saves_state_changes_through_the_repository():
    repository = RecordingRepository()
    task = _task()

    _harness(Gateway(), repository=repository).run(task)

    assert repository.get(task.id) is task
    assert repository.states[0] is TaskState.TASK_RUNNING
    assert repository.states[-1] is TaskState.TASK_COMPLETED
    assert task.state is TaskState.TASK_COMPLETED
    assert task.plan.goal == ["buy a phone"]
    assert task.plan.current == "compare"


def test_planning_run_emits_task_events():
    output = RecordingOutput()

    _harness(Gateway(), event_handler=output).run(_task())

    assert [event.type for event in output.events] == [
        EventType.TASK_STARTED,
        EventType.TASK_PLANNING,
        EventType.LLM_STARTED,
        EventType.LLM_COMPLETED,
        EventType.TASK_PLANNED,
        EventType.LLM_STARTED,
        EventType.LLM_DELTA,
        EventType.LLM_COMPLETED,
        EventType.TASK_COMPLETED,
    ]
    planned = output.events[4]
    assert planned.payload.goal == ["buy a phone"]
    llm = [event for event in output.events if event.type.name.startswith("LLM_")]
    assert [event.payload.phase for event in llm] == [
        "planning",
        "planning",
        "infer",
        "infer",
        "infer",
    ]


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
    assert [event.type for event in output.events if event.type.name.startswith("LLM_")] == [
        EventType.LLM_STARTED,
        EventType.LLM_RETRYING,
        EventType.LLM_STARTED,
        EventType.LLM_COMPLETED,
        EventType.LLM_STARTED,
        EventType.LLM_DELTA,
        EventType.LLM_COMPLETED,
    ]


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
    assert [event.type for event in output.events] == [
        EventType.LLM_STARTED,
        EventType.LLM_DELTA,
        EventType.LLM_DELTA,
        EventType.LLM_COMPLETED,
    ]
    assert [event.payload.content for event in output.events[1:3]] == ["hel", "lo"]
    assert output.events[0].payload.phase == "infer"


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
    assert [event.type for event in output.events] == [
        EventType.LLM_STARTED,
        EventType.LLM_COMPLETED,
    ]


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
        [],
        Executor([tool]),
        DefaultContextBuilder(),
        harness_guard=NoOpGuard(),
    )

    context = harness.build_planning_context(_task())

    assert context.tools == []
    assert "shell: Run a command." in context.messages[0].content


def test_context_reads_tools_from_the_executor_catalog_at_call_time():
    registry = DefaultToolRegistry()
    executor = DefaultToolExecutor(registry)
    harness = Harness(
        "Be brief.",
        Gateway(),
        [],
        executor,
        DefaultContextBuilder(),
        harness_guard=NoOpGuard(),
    )
    registry.register(_shell())

    context = harness.build_planning_context(_task())

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


def test_length_continues_once_with_tools_then_completes():
    tool = _shell()
    gateway = ScriptedGateway(
        [
            _model(FinishReason.LENGTH, "hel"),
            _model(FinishReason.STOP, "hello"),
        ]
    )

    assert _harness(gateway, tools=[tool]).run(_task()).output == "hello"

    assert len(gateway.infer_contexts) == 2
    assert gateway.infer_contexts[1].tools == [tool]


def test_execute_tool_calls_emits_start_and_completion():
    class Completing(Executor):
        def execute_tool_call(self, call, approved=False, hold=False):
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
                output="listed",
            )

    output = RecordingOutput()
    executor = Completing()
    harness = _harness(Gateway(), event_handler=output, executor=executor)
    task = _task()

    result = harness.execute_tool_calls(
        [LlmToolCall("call_0_abc", "shell", '{"command": "dir"}')],
        task,
    )

    assert result.completed_tools[0].output == "listed"
    assert [event.type for event in output.events] == [
        EventType.TOOL_EXECUTION_STARTED,
        EventType.TOOL_EXECUTION_COMPLETED,
    ]
    completed = output.events[1]
    assert completed.payload.name == "shell"
    assert completed.payload.arguments == '{"command": "dir"}'
    assert completed.payload.tool_call_id == "call_0_abc"
    assert completed.payload.content == "listed"
    assert completed.task_id == str(task.id)
    assert not hasattr(executor, "event_handler")


def test_execute_tool_calls_emits_failure():
    class Failing(Executor):
        def execute_tool_call(self, call, approved=False, hold=False):
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                output="RuntimeError: nope",
                fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_UNKNOWN,
            )

    output = RecordingOutput()
    harness = _harness(Gateway(), event_handler=output, executor=Failing())

    result = harness.execute_tool_calls(
        [LlmToolCall("call_1", "shell", "{}")],
        _task(),
    )

    assert result.failed_tools[0].output == "RuntimeError: nope"
    assert [event.type for event in output.events] == [
        EventType.TOOL_EXECUTION_STARTED,
        EventType.TOOL_EXECUTION_FAILED,
    ]
    assert output.events[1].payload.reason == "unknown"
    assert output.events[1].payload.content == "RuntimeError: nope"


def test_step_limit_asks_for_a_closing_answer_from_unseen_tool_results():
    executor = CountingExecutor([_shell()])
    gateway = ScriptedGateway(
        [
            _model(FinishReason.TOOL_CALLS, None, [LlmToolCall("call_1", "shell", "{}")]),
            _model(FinishReason.STOP, "based on the tool"),
        ]
    )
    task = _task()

    assert _harness(gateway, executor=executor, max_steps=1).run(task).output == "based on the tool"

    assert task.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS
    assert executor.batches == 1
    assert gateway.infer_contexts[1].tools == []
    assert task.history[-1].role == "tool"


def test_closing_answer_without_text_fails_with_an_empty_result():
    gateway = ScriptedGateway(
        [
            _model(FinishReason.TOOL_CALLS, None, [LlmToolCall("call_1", "shell", "{}")]),
            _model(FinishReason.TOOL_CALLS, None, [LlmToolCall("call_2", "shell", "{}")]),
        ]
    )
    task = _task()

    assert _harness(gateway, tools=[_shell()], max_steps=1).run(task).output == ""

    assert task.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS
    assert len(gateway.infer_contexts) == 2


def _recording_executor(called: list, *names: str):
    registry = DefaultToolRegistry()
    registry.tools = {}
    for name in names:
        registry.register(
            Tool(
                id=name,
                name=name,
                description="A tool.",
                input_schema={},
                executer=lambda inputs, name=name: called.append((name, dict(inputs))) or "ran",
            )
        )
    return DefaultToolExecutor(registry)


def _shell_call(call_id: str = "call_1") -> LLMOutput:
    return _model(
        FinishReason.TOOL_CALLS,
        None,
        [LlmToolCall(call_id, "shell", '{"command": "git status"}')],
    )


def _tool_messages(task: Task) -> list:
    return [message for message in task.history if message.role == "tool"]


def test_shell_call_pauses_without_running_or_a_closing_call():
    called = []
    output = RecordingOutput()
    gateway = ScriptedGateway([_shell_call(), _model(FinishReason.STOP, "should not run")])
    task = _task()

    result = _harness(
        gateway,
        event_handler=output,
        tools=[_shell()],
        executor=_recording_executor(called, "shell"),
    ).run(task)

    assert called == []
    assert result.output == ""
    assert task.state is TaskState.TASK_PAUSED
    assert task.state_reason is TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL
    assert task.pending_tool_calls[0].arguments == '{"command": "git status"}'
    assert len(gateway.infer_contexts) == 1
    paused = [e for e in output.events if e.type is EventType.TOOL_EXECUTION_PAUSED]
    assert [event.payload.tool_call_id for event in paused] == ["call_1"]


def test_approved_shell_runs_before_the_next_inference():
    called = []
    gateway = ScriptedGateway([_shell_call(), _model(FinishReason.STOP, "status read")])
    harness = _harness(gateway, executor=_recording_executor(called, "shell"))
    task = _task()
    harness.run(task)

    task.decide_tool_call("call_1", approved=True)
    result = harness.run(task)

    assert called == [("shell", {"command": "git status"})]
    assert result.output == "status read"
    assert task.state is TaskState.TASK_COMPLETED
    assert task.pending_tool_calls == []
    assert [message.content for message in _tool_messages(task)] == ["ran"]


def test_rejected_shell_is_recorded_before_the_next_inference():
    called = []
    gateway = ScriptedGateway([_shell_call(), _model(FinishReason.STOP, "skipped")])
    harness = _harness(gateway, executor=_recording_executor(called, "shell"))
    task = _task()
    harness.run(task)

    task.decide_tool_call("call_1", approved=False)
    result = harness.run(task)

    assert called == []
    assert result.output == "skipped"
    assert [message.content for message in _tool_messages(task)] == ["The tool call was rejected."]


def test_calls_after_a_held_call_wait_and_run_in_order():
    called = []
    gateway = ScriptedGateway(
        [
            _model(
                FinishReason.TOOL_CALLS,
                None,
                [
                    LlmToolCall("call_1", "read_file", '{"path": "a"}'),
                    LlmToolCall("call_2", "shell", '{"command": "git status"}'),
                    LlmToolCall("call_3", "read_file", '{"path": "b"}'),
                ],
            ),
            _model(FinishReason.STOP, "done"),
        ]
    )
    harness = _harness(gateway, executor=_recording_executor(called, "shell", "read_file"))
    task = _task()
    harness.run(task)

    assert called == [("read_file", {"path": "a"})]
    assert [item.tool_id for item in task.approval_requests()] == ["call_2"]
    messages = {message.tool_call_id: message.content for message in _tool_messages(task)}
    assert messages == {
        "call_1": "ran",
        "call_2": PENDING_TOOL_MESSAGE,
        "call_3": HELD_TOOL_MESSAGE,
    }

    task.decide_tool_call("call_2", approved=True)
    harness.run(task)

    assert called == [
        ("read_file", {"path": "a"}),
        ("shell", {"command": "git status"}),
        ("read_file", {"path": "b"}),
    ]
    assert task.state is TaskState.TASK_COMPLETED


def test_each_held_call_follows_its_own_decision():
    called = []
    gateway = ScriptedGateway(
        [
            _model(
                FinishReason.TOOL_CALLS,
                None,
                [
                    LlmToolCall("call_1", "shell", '{"command": "git status"}'),
                    LlmToolCall("call_2", "write_file", '{"path": "x.txt"}'),
                ],
            ),
            _model(FinishReason.STOP, "done"),
        ]
    )
    harness = _harness(gateway, executor=_recording_executor(called, "shell", "write_file"))
    task = _task()
    harness.run(task)

    assert [item.tool_id for item in task.approval_requests()] == ["call_1", "call_2"]

    task.decide_tool_call("call_1", approved=True)
    with pytest.raises(ValueError):
        harness.run(task)

    task.decide_tool_call("call_2", approved=False)
    harness.run(task)

    assert called == [("shell", {"command": "git status"})]
    assert [message.content for message in _tool_messages(task)] == [
        "ran",
        "The tool call was rejected.",
    ]


def test_paused_task_with_undecided_calls_does_not_run():
    gateway = ScriptedGateway([_shell_call()])
    harness = _harness(gateway, executor=_recording_executor([], "shell"))
    task = _task()
    harness.run(task)

    with pytest.raises(ValueError):
        harness.run(task)

    assert task.state is TaskState.TASK_PAUSED
    assert len(task.pending_tool_calls) == 1


def test_failure_after_approval_fails_the_task():
    class Failing(ScriptedGateway):
        def infer(self, context, stream=True, response_format=None):
            if len(self.infer_contexts) == 1:
                self.infer_contexts.append(context)
                raise ValueError("bad request")
            return super().infer(context, stream, response_format)

    gateway = Failing([_shell_call()])
    harness = _harness(gateway, executor=_recording_executor([], "shell"))
    task = _task()
    harness.run(task)

    task.decide_tool_call("call_1", approved=False)
    harness.run(task)

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_LLM_ERROR
    assert task.pending_tool_calls == []


def test_step_limit_counts_steps_taken_before_the_approval():
    gateway = ScriptedGateway(
        [
            _shell_call(),
            _model(FinishReason.LENGTH, "part"),
            _model(FinishReason.STOP, "closing"),
        ]
    )
    harness = _harness(gateway, executor=_recording_executor([], "shell"), max_steps=2)
    task = _task()
    harness.run(task)

    task.decide_tool_call("call_1", approved=True)
    result = harness.run(task)

    assert task.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS
    assert result.output == "closing"


def test_clarification_completes_without_inference():
    class Clarifying(Gateway):
        def planning(self, context, stream=True, response_format=None):
            self.planning_calls.append((stream, response_format))
            return LLMOutput(
                FinishReason.STOP,
                None,
                [],
                None,
                None,
                structured=TaskPlan(goal=[], clarification="What should I do with 12?"),
            )

        def infer(self, context, stream=True, response_format=None):
            raise AssertionError("infer should not run")

    task = _task()
    output = RecordingOutput()

    result = _harness(Clarifying(), event_handler=output).run(task)

    assert result.output == "What should I do with 12?"
    assert task.state is TaskState.TASK_COMPLETED
    deltas = [e.payload.content for e in output.events if e.type is EventType.LLM_DELTA]
    assert deltas == ["What should I do with 12?"]
