import asyncio
import threading
from typing import get_type_hints

import pytest
from src.context.context_builder import DefaultContextBuilder
from src.core.cancellation import CancellationToken
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
from src.persistence.in_memory import InMemoryTaskRepository
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
    ToolCallResult,
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


class FalseyEventSink(RecordingOutput):
    def __bool__(self):
        return False


def test_harness_preserves_falsey_event_sink():
    sink = FalseyEventSink()
    harness = _harness(Gateway(), event_handler=sink)

    harness.run(_task())

    assert harness.event_handler is sink
    assert sink.events


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


def test_completed_turn_persists_result_and_progress_snapshot():
    repository = InMemoryTaskRepository()
    task = _task()

    result = _harness(Gateway(), repository=repository).run(task)
    restored = repository.get(task.id)

    assert result is task.result
    assert task.step_count == 1
    assert task.elapsed_seconds >= 0
    assert restored is not None
    assert restored.state is TaskState.TASK_COMPLETED
    assert restored.result == result
    assert restored.step_count == 1


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
    async def no_cancel(_self, _seconds):
        return False

    monkeypatch.setattr(CancellationToken, "await_cancelled", no_cancel)
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
        EventType.LLM_FAILED,
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


def test_transient_planning_errors_are_retried(monkeypatch):
    waits = []

    async def record_wait(_token, seconds):
        waits.append(seconds)
        return False

    monkeypatch.setattr(CancellationToken, "await_cancelled", record_wait)
    gateway = Gateway()
    gateway.failures = 1

    assert _harness(gateway).run(_task()).output == "bought"

    assert waits == [DEFAULT_LLM_ATTEMPT_DELAY_SECONDS]
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


def test_run_declares_a_non_nullable_result():
    assert get_type_hints(Harness.run)["return"] is TaskResult


def test_completed_turn_normalizes_missing_model_content():
    gateway = ScriptedGateway([_model(FinishReason.STOP)])
    task = _task()

    result = _harness(gateway).run(task)

    assert result is not None
    assert result.output == ""
    assert task.result is result


def test_finished_task_returns_its_result():
    task = _task()
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    task.set_result(TaskResult("bought"))
    task.transition_to_completed()

    assert _harness(Gateway()).run(task).output == "bought"
    assert task.state is TaskState.TASK_COMPLETED


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


def test_executor_error_after_start_emits_a_matching_tool_failure():
    class RaisesAfterStart(Executor):
        def execute_tool_calls(
            self,
            tool_calls,
            *,
            before_call=None,
            after_call=None,
            should_cancel=None,
        ):
            del after_call, should_cancel
            before_call(tool_calls[0])
            raise RuntimeError("executor crashed")

    output = RecordingOutput()
    harness = _harness(Gateway(), event_handler=output, executor=RaisesAfterStart())

    with pytest.raises(RuntimeError, match="executor crashed"):
        harness.execute_tool_calls([LlmToolCall("call_1", "shell", "{}")], _task())

    assert [event.type for event in output.events] == [
        EventType.TOOL_EXECUTION_STARTED,
        EventType.TOOL_EXECUTION_FAILED,
    ]
    assert output.events[-1].payload.reason == "unknown"


def test_step_limit_asks_for_a_closing_answer_from_unseen_tool_results():
    executor = CountingExecutor([_shell()])
    gateway = ScriptedGateway(
        [
            _model(FinishReason.TOOL_CALLS, None, [LlmToolCall("call_1", "shell", "{}")]),
            _model(FinishReason.STOP, "based on the tool"),
        ]
    )
    task = _task()
    repository = InMemoryTaskRepository()

    result = _harness(gateway, executor=executor, max_steps=1, repository=repository).run(task)

    restored = repository.get(task.id)
    assert result.output == "based on the tool"
    assert result is task.result
    assert task.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS
    assert task.step_count == 1
    assert restored is not None
    assert restored.result == result
    assert restored.step_count == 1
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
    assert task.pending_tool_calls[0].arguments == '{"command":"git status"}'
    assert len(gateway.infer_contexts) == 1
    paused = [e for e in output.events if e.type is EventType.TOOL_EXECUTION_PAUSED]
    assert [event.payload.tool_call_id for event in paused] == ["call_1"]


def test_precancelled_created_task_does_no_gateway_work():
    gateway = Gateway()
    repository = InMemoryTaskRepository()
    token = CancellationToken()
    token.cancel()
    task = _task()

    result = _harness(gateway, repository=repository).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert gateway.planning_calls == []
    assert repository.get(task.id) == task


def test_precancelled_paused_task_cancels_held_calls_in_order():
    called = []

    class FailChildCancellation(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_FAILED:
                raise RuntimeError("child terminal failed")

    sink = FailChildCancellation()
    repository = InMemoryTaskRepository()
    gateway = ScriptedGateway([_shell_call()])
    harness = _harness(
        gateway,
        event_handler=sink,
        executor=_recording_executor(called, "shell"),
        repository=repository,
    )
    task = _task()
    harness.run(task)
    token = CancellationToken()
    token.cancel()
    sink.events.clear()

    result = harness.run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert task.pending_tool_calls == []
    assert called == []
    assert [message.content for message in _tool_messages(task)] == ["Tool call cancelled."]
    assert [event.type for event in sink.events] == [
        EventType.TOOL_EXECUTION_FAILED,
        EventType.TASK_FAILED,
    ]
    assert repository.get(task.id) == task


def test_cancel_between_held_calls_stops_the_remaining_calls():
    token = CancellationToken()
    called = []

    def run(inputs):
        called.append(inputs)
        token.cancel()
        return "ran"

    registry = DefaultToolRegistry()
    registry.register(Tool("shell", "shell", "Run.", {}, run))

    class FailChildCancellation(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_FAILED:
                raise RuntimeError("child terminal failed")

    harness = _harness(
        ScriptedGateway(
            [
                _model(
                    FinishReason.TOOL_CALLS,
                    None,
                    [
                        LlmToolCall("call-1", "shell", '{"command":"first"}'),
                        LlmToolCall("call-2", "shell", '{"command":"second"}'),
                    ],
                )
            ]
        ),
        executor=DefaultToolExecutor(registry),
        event_handler=FailChildCancellation(),
    )
    task = _task()
    harness.run(task)
    task.decide_tool_call("call-1", approved=True)
    task.decide_tool_call("call-2", approved=True)

    result = harness.run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert called == [{"command": "first"}]
    assert [message.content for message in _tool_messages(task)] == [
        "ran",
        "Tool call cancelled.",
    ]


def test_cancel_during_planning_return_wins_before_plan_is_saved():
    token = CancellationToken()

    class FailChildCancellation(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.LLM_FAILED:
                raise RuntimeError("child terminal failed")

    sink = FailChildCancellation()

    class CancellingPlanning(Gateway):
        def planning(self, context, stream=True, response_format=None):
            token.cancel()
            return super().planning(context, stream, response_format)

    task = _task()
    result = _harness(CancellingPlanning(), event_handler=sink).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert task.plan is None
    assert [
        (event.type, event.payload.reason)
        for event in sink.events
        if event.type in {EventType.LLM_COMPLETED, EventType.LLM_FAILED}
    ] == [(EventType.LLM_FAILED, "cancel")]


def test_cancel_during_gateway_error_wins_over_llm_error():
    token = CancellationToken()

    class CancellingError(Gateway):
        def planning(self, context, stream=True, response_format=None):
            token.cancel()
            raise ValueError("gateway failed")

    task = _task()
    result = _harness(CancellingError()).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL


def test_cancel_interrupts_retry_backoff_without_another_attempt():
    class CancellingWaitToken(CancellationToken):
        def __init__(self):
            super().__init__()
            self.waits = []

        async def await_cancelled(self, timeout=None):
            self.waits.append(timeout)
            self.cancel()
            return True

    token = CancellingWaitToken()
    gateway = Gateway()
    gateway.failures = 1
    task = _task()
    sink = RecordingOutput()

    result = _harness(gateway, event_handler=sink).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert token.waits == [DEFAULT_LLM_ATTEMPT_DELAY_SECONDS]
    assert len(gateway.planning_calls) == 1
    assert [event.type for event in sink.events if event.type.name.startswith("LLM_")] == [
        EventType.LLM_STARTED,
        EventType.LLM_FAILED,
        EventType.LLM_RETRYING,
    ]


def test_cancel_during_inference_prevents_tool_execution_and_completion():
    token = CancellationToken()
    called = []

    class CancellingInference(Gateway):
        def infer(self, context, stream=True, response_format=None):
            token.cancel()
            return _shell_call()

    task = _task()
    result = _harness(CancellingInference(), executor=_recording_executor(called, "shell")).run(
        task, cancellation_token=token
    )

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert called == []
    assert task.pending_tool_calls == []


def test_cancel_during_limit_finalization_emits_only_cancel_failure():
    token = CancellationToken()
    sink = RecordingOutput()

    class CancellingFinalization(Gateway):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def infer(self, context, stream=True, response_format=None):
            self.calls += 1
            if self.calls == 1:
                return _model(FinishReason.LENGTH, "partial")
            token.cancel()
            return _model(FinishReason.STOP, "closing")

    task = _task()
    result = _harness(CancellingFinalization(), event_handler=sink, max_steps=1).run(
        task, cancellation_token=token
    )

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    failed = [event for event in sink.events if event.type is EventType.TASK_FAILED]
    assert [event.payload.reason for event in failed] == ["cancel"]


def test_cancel_from_finalization_completed_callback_wins_over_limit_failure():
    token = CancellationToken()

    class CancelSecondCompletion(RecordingOutput):
        def __init__(self):
            super().__init__()
            self.completions = 0

        def emit(self, event):
            super().emit(event)
            if event.type is EventType.LLM_COMPLETED and event.payload.phase == "infer":
                self.completions += 1
                if self.completions == 2:
                    token.cancel()

    task = _task()
    sink = CancelSecondCompletion()
    result = _harness(
        ScriptedGateway(
            [_model(FinishReason.LENGTH, "partial"), _model(FinishReason.STOP, "closing")]
        ),
        event_handler=sink,
        max_steps=1,
    ).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert [
        event.payload.reason for event in sink.events if event.type is EventType.TASK_FAILED
    ] == ["cancel"]


def test_cancel_from_planning_completed_callback_prevents_clarification_completion():
    token = CancellationToken()

    class ClarifyingGateway(Gateway):
        def planning(self, context, stream=True, response_format=None):
            return LLMOutput(
                FinishReason.STOP,
                None,
                [],
                None,
                None,
                structured=TaskPlan(goal=[], clarification="Which target?"),
            )

    class CancelPlanningCompletion(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.LLM_COMPLETED and event.payload.phase == "planning":
                token.cancel()

    task = _task()
    result = _harness(ClarifyingGateway(), event_handler=CancelPlanningCompletion()).run(
        task, cancellation_token=token
    )

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL


def test_cancelled_inference_does_not_publish_abandoned_answer_delta():
    token = CancellationToken()
    sink = RecordingOutput()

    class CancellingAnswer(Gateway):
        def infer(self, context, stream=True, response_format=None):
            token.cancel()
            return _model(FinishReason.STOP, "should-not-publish")

    task = _task()
    _harness(CancellingAnswer(), event_handler=sink).run(task, cancellation_token=token)

    assert all(
        not (event.type is EventType.LLM_DELTA and event.payload.content == "should-not-publish")
        for event in sink.events
    )


def test_uncancelled_run_supports_legacy_tool_executor_batch_signature():
    class LegacyExecutor(Executor):
        def __init__(self, tools=None):
            super().__init__(tools)
            self.legacy_batches = 0

        def execute_tool_calls(self, tool_calls, *, before_call=None, after_call=None):
            self.legacy_batches += 1
            return super().execute_tool_calls(
                tool_calls,
                before_call=before_call,
                after_call=after_call,
            )

    tool = Tool("read_file", "read_file", "Read.", {}, lambda inputs: "read")
    gateway = ScriptedGateway(
        [
            _model(
                FinishReason.TOOL_CALLS,
                None,
                [LlmToolCall("call-1", "read_file", '{"path":"a"}')],
            ),
            _model(FinishReason.STOP, "done"),
        ]
    )
    task = _task()

    executor = LegacyExecutor([tool])
    result = _harness(gateway, executor=executor).run(task)

    assert result == TaskResult(output="done")
    assert task.state is TaskState.TASK_COMPLETED
    assert executor.legacy_batches == 1


def test_cancel_from_llm_started_callback_prevents_gateway_call():
    token = CancellationToken()

    class CancelOnStart(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.LLM_STARTED and event.payload.phase == "planning":
                token.cancel()

    gateway = Gateway()
    task = _task()

    result = _harness(gateway, event_handler=CancelOnStart()).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert gateway.planning_calls == []


def test_cancel_from_tool_started_callback_prevents_default_tool_execution():
    token = CancellationToken()
    called = []

    class CancelOnToolStart(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_STARTED:
                token.cancel()
            if event.type is EventType.TOOL_EXECUTION_FAILED:
                raise RuntimeError("terminal sink failed")

    gateway = ScriptedGateway(
        [
            _model(
                FinishReason.TOOL_CALLS,
                calls=[LlmToolCall("call-1", "read_file", "{}")],
            )
        ]
    )
    sink = CancelOnToolStart()
    task = _task()

    result = _harness(
        gateway,
        event_handler=sink,
        executor=_recording_executor(called, "read_file"),
    ).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert called == []
    assert [event.type for event in sink.events if event.type.name.startswith("TOOL_")] == [
        EventType.TOOL_EXECUTION_STARTED,
        EventType.TOOL_EXECUTION_FAILED,
    ]


def test_public_tool_batch_returns_cancelled_result_when_start_sink_cancels():
    token = CancellationToken()
    called = []

    class CancelOnStart(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_STARTED:
                token.cancel()

    harness = _harness(
        Gateway(),
        event_handler=CancelOnStart(),
        executor=_recording_executor(called, "read_file"),
    )

    result = harness.execute_tool_calls(
        [LlmToolCall("call-1", "read_file", "{}")],
        _task(),
        cancellation_token=token,
    )

    assert called == []
    assert result.completed_tools == []
    assert [item.tool_id for item in result.failed_tools] == ["call-1"]
    assert result.failed_tools[0].fail_reason is (
        ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED
    )


def test_terminal_sink_failure_after_tool_cancels_does_not_skip_task_cleanup():
    token = CancellationToken()

    def cancel_during_tool(_inputs):
        token.cancel()
        return "abandoned"

    class FailCancelledToolTerminal(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_FAILED:
                raise RuntimeError("terminal sink failed")

    registry = DefaultToolRegistry()
    registry.register(Tool("read_file", "read_file", "Read.", {}, cancel_during_tool))
    task = _task()

    result = _harness(
        ScriptedGateway(
            [
                _model(
                    FinishReason.TOOL_CALLS,
                    calls=[LlmToolCall("call-1", "read_file", "{}")],
                )
            ]
        ),
        event_handler=FailCancelledToolTerminal(),
        executor=DefaultToolExecutor(registry),
    ).run(task, cancellation_token=token)

    assert result == TaskResult(output="Task cancelled.")
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL


def test_repository_failure_while_cancelling_still_propagates():
    class FailingRepository(InMemoryTaskRepository):
        def save(self, task):
            if task.is_failed():
                raise OSError("cancel snapshot failed")
            super().save(task)

    token = CancellationToken()
    token.cancel()

    with pytest.raises(OSError, match="cancel snapshot failed"):
        _harness(Gateway(), repository=FailingRepository()).run(_task(), cancellation_token=token)


def test_event_failure_while_cancelling_still_propagates():
    class FailingSink(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TASK_FAILED:
                raise RuntimeError("cancel observer failed")

    token = CancellationToken()
    token.cancel()

    with pytest.raises(RuntimeError, match="cancel observer failed"):
        _harness(Gateway(), event_handler=FailingSink()).run(_task(), cancellation_token=token)


def test_finished_task_ignores_a_cancelled_token():
    task = Task(
        "finished",
        state=TaskState.TASK_COMPLETED,
        result=TaskResult(output="done"),
    )
    token = CancellationToken()
    token.cancel()
    gateway = Gateway()

    result = _harness(gateway).run(task, cancellation_token=token)

    assert result == TaskResult(output="done")
    assert task.state is TaskState.TASK_COMPLETED
    assert gateway.planning_calls == []


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
    repository = InMemoryTaskRepository()

    result = _harness(Clarifying(), event_handler=output, repository=repository).run(task)

    assert result.output == "What should I do with 12?"
    assert task.state is TaskState.TASK_COMPLETED
    restored = repository.get(task.id)
    assert restored is not None
    assert restored.state is TaskState.TASK_COMPLETED
    assert restored.result == result
    deltas = [e.payload.content for e in output.events if e.type is EventType.LLM_DELTA]
    assert deltas == ["What should I do with 12?"]


def test_unexpected_error_persists_the_returned_result():
    class BrokenPlanning(Gateway):
        def planning(self, context, stream=True, response_format=None):
            raise ValueError("bad request")

    repository = InMemoryTaskRepository()
    task = _task()
    output = RecordingOutput()

    result = _harness(BrokenPlanning(), repository=repository, event_handler=output).run(task)
    restored = repository.get(task.id)

    assert result is task.result
    assert result.output == "An unexpected error occurred: bad request"
    assert restored is not None
    assert restored.result == result
    llm_events = [event for event in output.events if event.type.name.startswith("LLM_")]
    assert [event.type for event in llm_events] == [
        EventType.LLM_STARTED,
        EventType.LLM_FAILED,
    ]
    assert llm_events[-1].payload.reason == "ValueError"


def test_finalization_error_replaces_the_empty_failed_result():
    class FinalizationFails(Gateway):
        def __init__(self):
            super().__init__()
            self.infer_calls = 0

        def infer(self, context, stream=True, response_format=None):
            self.infer_calls += 1
            if self.infer_calls == 1:
                return _model(FinishReason.LENGTH, "partial")
            raise ValueError("closing failed")

    repository = InMemoryTaskRepository()
    task = _task()

    result = _harness(FinalizationFails(), max_steps=1, repository=repository).run(task)
    restored = repository.get(task.id)

    assert result is task.result
    assert result.output == "An unexpected error occurred: closing failed"
    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS
    assert restored is not None
    assert restored.result == result
    assert restored.state_reason is TaskFailReason.TASK_FAILED_MAX_STEPS


def test_terminal_save_error_is_not_rewritten_as_a_task_failure():
    class FailingTerminalRepository:
        def __init__(self):
            self.inner = InMemoryTaskRepository()

        def save(self, task):
            if task.is_finished():
                raise OSError("terminal storage unavailable")
            self.inner.save(task)

        def get(self, task_id):
            return self.inner.get(task_id)

    repository = FailingTerminalRepository()
    task = _task()

    with pytest.raises(OSError, match="terminal storage unavailable"):
        _harness(Gateway(), repository=repository).run(task)

    assert task.state is TaskState.TASK_COMPLETED
    assert task.result == TaskResult(output="bought")
    stored = repository.get(task.id)
    assert stored is not None
    assert stored.state is TaskState.TASK_RUNNING


def test_terminal_event_error_preserves_the_committed_result():
    class FailingCompletedSink(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TASK_COMPLETED:
                raise RuntimeError("completed observer failed")

    repository = InMemoryTaskRepository()
    task = _task()

    with pytest.raises(RuntimeError, match="completed observer failed"):
        _harness(Gateway(), event_handler=FailingCompletedSink(), repository=repository).run(task)

    stored = repository.get(task.id)
    assert task.state is TaskState.TASK_COMPLETED
    assert task.result == TaskResult(output="bought")
    assert stored is not None
    assert stored.state is TaskState.TASK_COMPLETED
    assert stored.result == TaskResult(output="bought")


def test_planning_completion_event_error_escapes_retry_handling():
    class FailingPlanningSink(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.LLM_COMPLETED and event.payload.phase == "planning":
                raise RuntimeError("planning observer failed")

    task = _task()

    with pytest.raises(RuntimeError, match="planning observer failed"):
        _harness(Gateway(), event_handler=FailingPlanningSink()).run(task)

    assert task.state is TaskState.TASK_RUNNING
    assert task.result is None


def test_inference_delta_event_error_escapes_retry_handling():
    class FailingDeltaSink(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.LLM_DELTA and event.payload.phase == "infer":
                raise RuntimeError("delta observer failed")

    task = _task()

    with pytest.raises(RuntimeError, match="delta observer failed"):
        _harness(Gateway(), event_handler=FailingDeltaSink()).run(task)

    assert task.state is TaskState.TASK_RUNNING
    assert task.result is None


def test_error_snapshot_failure_preserves_the_repository_exception():
    class FailingFailureRepository(InMemoryTaskRepository):
        def save(self, task):
            if task.is_failed():
                raise OSError("failure snapshot unavailable")
            super().save(task)

    class BrokenPlanning(Gateway):
        def planning(self, context, stream=True, response_format=None):
            raise ValueError("bad request")

    task = _task()

    with pytest.raises(OSError, match="failure snapshot unavailable"):
        _harness(BrokenPlanning(), repository=FailingFailureRepository()).run(task)

    assert task.state is TaskState.TASK_FAILED
    assert task.result == TaskResult(output="An unexpected error occurred: bad request")


def test_failed_event_never_exposes_an_empty_terminal_snapshot():
    repository = InMemoryTaskRepository()

    class InspectingFailureSink(RecordingOutput):
        observed = None

        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TASK_FAILED:
                stored = repository.get(event.task_id)
                assert stored is not None
                self.observed = (stored.state, stored.result)

    sink = InspectingFailureSink()
    gateway = ScriptedGateway(
        [_model(FinishReason.LENGTH, "partial"), _model(FinishReason.STOP, "closing")]
    )
    task = _task()

    result = _harness(
        gateway,
        event_handler=sink,
        max_steps=1,
        repository=repository,
    ).run(task)

    assert sink.observed == (TaskState.TASK_FAILED, TaskResult(output="closing"))
    assert result.output == "closing"
    stored = repository.get(task.id)
    assert stored is not None
    assert stored.state is TaskState.TASK_FAILED
    assert stored.result == result


class FakeClock:
    def __init__(self, value=100.0):
        self.value = value

    def monotonic(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_clarification_elapsed_time_includes_planning(monkeypatch):
    clock = FakeClock()

    class Clarifying(Gateway):
        def planning(self, context, stream=True, response_format=None):
            clock.advance(4)
            return LLMOutput(
                FinishReason.STOP,
                None,
                [],
                None,
                None,
                structured=TaskPlan(goal=[], clarification="Which target?"),
            )

    monkeypatch.setattr("src.core.loop.time.monotonic", clock.monotonic)
    task = _task()

    _harness(Clarifying()).run(task)

    assert task.elapsed_seconds == 4.0


def test_failed_inference_elapsed_time_includes_planning_and_error(monkeypatch):
    clock = FakeClock()

    class FailingInference(Gateway):
        def planning(self, context, stream=True, response_format=None):
            clock.advance(2)
            return super().planning(context, stream, response_format)

        def infer(self, context, stream=True, response_format=None):
            clock.advance(3)
            raise ValueError("bad request")

    monkeypatch.setattr("src.core.loop.time.monotonic", clock.monotonic)
    task = _task()

    _harness(FailingInference()).run(task)

    assert task.elapsed_seconds == 5.0


def test_failed_turn_elapsed_time_includes_successful_finalization(monkeypatch):
    clock = FakeClock()

    class TimedFinalization(ScriptedGateway):
        def planning(self, context, stream=True, response_format=None):
            clock.advance(2)
            return super().planning(context, stream, response_format)

        def infer(self, context, stream=True, response_format=None):
            clock.advance(3 if not self.infer_contexts else 4)
            return super().infer(context, stream, response_format)

    gateway = TimedFinalization(
        [_model(FinishReason.LENGTH, "partial"), _model(FinishReason.STOP, "closing")]
    )
    monkeypatch.setattr("src.core.loop.time.monotonic", clock.monotonic)
    task = _task()

    _harness(gateway, max_steps=1).run(task)

    assert task.elapsed_seconds == 9.0


def test_failed_turn_elapsed_time_includes_finalization_error(monkeypatch):
    clock = FakeClock()

    class TimedFinalizationFailure(Gateway):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def planning(self, context, stream=True, response_format=None):
            clock.advance(2)
            return super().planning(context, stream, response_format)

        def infer(self, context, stream=True, response_format=None):
            self.calls += 1
            clock.advance(3 if self.calls == 1 else 4)
            if self.calls == 1:
                return _model(FinishReason.LENGTH, "partial")
            raise ValueError("closing failed")

    monkeypatch.setattr("src.core.loop.time.monotonic", clock.monotonic)
    task = _task()

    _harness(TimedFinalizationFailure(), max_steps=1).run(task)

    assert task.elapsed_seconds == 9.0


class NativeAsyncGateway:
    def __init__(self, answers=None):
        self.answers = list(answers or [_model(FinishReason.STOP, "async answer")])

    async def aplanning(self, context, stream=True, response_format=None):
        await asyncio.sleep(0)
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlan(goal=["async"], current="run", remaining=[]),
        )

    async def ainfer(self, context, stream=True, response_format=None):
        await asyncio.sleep(0)
        return self.answers.pop(0)

    def planning(self, context, stream=True, response_format=None):
        raise AssertionError("native async planning must be preferred")

    def infer(self, context, stream=True, response_format=None):
        raise AssertionError("native async inference must be preferred")


def test_arun_completes_with_a_native_async_gateway():
    task = _task()

    result = asyncio.run(_harness(NativeAsyncGateway()).arun(task))

    assert result.output == "async answer"
    assert task.state is TaskState.TASK_COMPLETED
    assert task.plan.goal == ["async"]


def test_arun_accepts_a_gateway_that_implements_only_the_async_protocol():
    class AsyncOnlyGateway:
        async def aplanning(self, context, stream=True, response_format=None):
            return LLMOutput(
                FinishReason.STOP,
                None,
                [],
                None,
                None,
                structured=TaskPlan(goal=["async only"]),
            )

        async def ainfer(self, context, stream=True, response_format=None):
            return _model(FinishReason.STOP, "done")

    result = asyncio.run(_harness(AsyncOnlyGateway()).arun(_task()))

    assert result.output == "done"


def test_arun_executes_async_tools_and_preserves_pause_resume_flow():
    called = []

    async def execute(inputs):
        called.append(inputs)
        return "tool output"

    gateway = NativeAsyncGateway(
        [
            _model(
                FinishReason.TOOL_CALLS,
                calls=[LlmToolCall("call-1", "shell", '{"command":"status"}')],
            ),
            _model(FinishReason.STOP, "done"),
        ]
    )
    registry = DefaultToolRegistry()
    registry.register(Tool("shell", "shell", "Run.", {}, execute))
    harness = _harness(gateway, executor=DefaultToolExecutor(registry))
    task = _task()

    asyncio.run(harness.arun(task))
    assert task.state is TaskState.TASK_PAUSED
    task.decide_tool_call("call-1", True)
    result = asyncio.run(harness.arun(task))

    assert result.output == "done"
    assert called == [{"command": "status"}]


def test_two_arun_calls_can_wait_in_the_same_async_gateway_concurrently():
    class RendezvousGateway(NativeAsyncGateway):
        def __init__(self):
            super().__init__()
            self.infer_count = 0
            self.both_started = asyncio.Event()

        async def ainfer(self, context, stream=True, response_format=None):
            self.infer_count += 1
            if self.infer_count == 2:
                self.both_started.set()
            await self.both_started.wait()
            return _model(FinishReason.STOP, "done")

    async def scenario():
        gateway = RendezvousGateway()
        harness = _harness(gateway)
        return await asyncio.wait_for(
            asyncio.gather(harness.arun(_task()), harness.arun(_task())), timeout=1
        )

    results = asyncio.run(scenario())

    assert [result.output for result in results] == ["done", "done"]


def test_sync_run_rejects_an_active_loop_before_emitting_or_saving():
    sink = RecordingOutput()
    repository = RecordingRepository()
    harness = _harness(Gateway(), event_handler=sink, repository=repository)

    async def scenario():
        with pytest.raises(RuntimeError, match="await arun"):
            harness.run(_task())

    asyncio.run(scenario())

    assert sink.events == []
    assert repository.states == []


def test_cancelling_the_arun_task_persists_cancel_and_reraises():
    class WaitingGateway(NativeAsyncGateway):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def ainfer(self, context, stream=True, response_format=None):
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario():
        gateway = WaitingGateway()
        repository = InMemoryTaskRepository()
        task = _task()
        running = asyncio.create_task(_harness(gateway, repository=repository).arun(task))
        await gateway.started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        return task, repository.get(task.id)

    task, restored = asyncio.run(scenario())

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert restored is not None
    assert restored.state_reason is TaskFailReason.TASK_FAILED_CANCEL


def test_child_observer_failure_during_caller_cancel_does_not_skip_task_cleanup():
    class WaitingGateway(NativeAsyncGateway):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def ainfer(self, context, stream=True, response_format=None):
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class FailChildTerminal(RecordingOutput):
        def emit(self, event):
            if event.type is EventType.LLM_FAILED:
                raise RuntimeError("child observer failed")
            super().emit(event)

    async def scenario():
        gateway = WaitingGateway()
        repository = InMemoryTaskRepository()
        task = _task()
        running = asyncio.create_task(
            _harness(
                gateway,
                event_handler=FailChildTerminal(),
                repository=repository,
            ).arun(task)
        )
        await gateway.started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        return task, repository.get(task.id)

    task, restored = asyncio.run(scenario())

    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert restored is not None
    assert restored.state_reason is TaskFailReason.TASK_FAILED_CANCEL


def test_cancellation_cleanup_repository_failure_wins_over_cancelled_error():
    class WaitingGateway(NativeAsyncGateway):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def ainfer(self, context, stream=True, response_format=None):
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class FailCancelledSave(RecordingRepository):
        def save(self, task):
            if task.state_reason is TaskFailReason.TASK_FAILED_CANCEL:
                raise RuntimeError("cancel save failed")
            super().save(task)

    async def scenario():
        gateway = WaitingGateway()
        task = _task()
        running = asyncio.create_task(_harness(gateway, repository=FailCancelledSave()).arun(task))
        await gateway.started.wait()
        running.cancel()
        with pytest.raises(RuntimeError, match="cancel save failed"):
            await running

    asyncio.run(scenario())


def test_cancellation_cleanup_error_keeps_cancelled_error_as_its_cause():
    class WaitingGateway(NativeAsyncGateway):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def ainfer(self, context, stream=True, response_format=None):
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class FailCancelledSave(RecordingRepository):
        def save(self, task):
            if task.state_reason is TaskFailReason.TASK_FAILED_CANCEL:
                raise RuntimeError("cancel save failed")
            super().save(task)

    async def scenario():
        gateway = WaitingGateway()
        running = asyncio.create_task(
            _harness(gateway, repository=FailCancelledSave()).arun(_task())
        )
        await gateway.started.wait()
        running.cancel()
        with pytest.raises(RuntimeError) as caught:
            await running
        return caught.value

    error = asyncio.run(scenario())

    assert isinstance(error.__cause__, asyncio.CancelledError)


def test_cancelled_legacy_batch_cannot_run_later_tools_or_publish_late_events():
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    executed = []

    class FailToolTerminal(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_FAILED:
                raise RuntimeError("tool observer failed")

    sink = FailToolTerminal()

    class BlockingLegacyExecutor(Executor):
        def execute_tool_calls(
            self,
            tool_calls,
            *,
            before_call=None,
            after_call=None,
            should_cancel=None,
        ):
            ordered = []
            try:
                for call in tool_calls:
                    if should_cancel is not None and should_cancel():
                        break
                    if before_call is not None:
                        before_call(call)
                    executed.append(call.id)
                    if len(executed) == 1:
                        started.set()
                        release.wait(1)
                    result = ToolExecutionResult(
                        tool_id=call.id,
                        tool_name=call.name,
                        state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
                        output="done",
                    )
                    ordered.append(result)
                    if after_call is not None:
                        after_call(call, result)
            finally:
                finished.set()
            return ToolCallResult(completed_tools=ordered, ordered_tools=ordered)

    gateway = NativeAsyncGateway(
        [
            _model(
                FinishReason.TOOL_CALLS,
                calls=[
                    LlmToolCall("call-1", "first", "{}"),
                    LlmToolCall("call-2", "second", "{}"),
                ],
            )
        ]
    )
    tools = [
        Tool("first", "first", "First.", {}, lambda inputs: "first"),
        Tool("second", "second", "Second.", {}, lambda inputs: "second"),
    ]
    harness = _harness(
        gateway,
        executor=BlockingLegacyExecutor(tools),
        event_handler=sink,
    )

    async def scenario():
        running = asyncio.create_task(harness.arun(_task()))
        await asyncio.to_thread(started.wait, 1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        terminal_event_count = len(sink.events)
        release.set()
        await asyncio.to_thread(finished.wait, 1)
        return terminal_event_count

    terminal_event_count = asyncio.run(scenario())

    assert executed == ["call-1"]
    assert len(sink.events) == terminal_event_count
    tool_events = [event for event in sink.events if event.type.name.startswith("TOOL_EXECUTION_")]
    assert [event.type for event in tool_events] == [
        EventType.TOOL_EXECUTION_STARTED,
        EventType.TOOL_EXECUTION_FAILED,
    ]
    assert tool_events[-1].payload.reason == "cancelled"


def test_cancelling_worker_sink_does_not_block_the_event_loop_on_callback_lock():
    sink_waiting = threading.Event()
    acknowledge = threading.Event()
    executor_finished = threading.Event()
    sink_timeouts = []
    executed = []

    class AcknowledgedSink(RecordingOutput):
        def emit(self, event):
            super().emit(event)
            if event.type is EventType.TOOL_EXECUTION_STARTED:
                sink_waiting.set()
                if not acknowledge.wait(1):
                    sink_timeouts.append(True)

    class SingleLegacyExecutor(Executor):
        def execute_tool_calls(
            self,
            tool_calls,
            *,
            before_call=None,
            after_call=None,
            should_cancel=None,
        ):
            del should_cancel
            try:
                call = tool_calls[0]
                before_call(call)
                executed.append(call.id)
                result = ToolExecutionResult(
                    tool_id=call.id,
                    tool_name=call.name,
                    state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
                    output="done",
                )
                after_call(call, result)
                return ToolCallResult(completed_tools=[result], ordered_tools=[result])
            finally:
                executor_finished.set()

    gateway = NativeAsyncGateway(
        [_model(FinishReason.TOOL_CALLS, calls=[LlmToolCall("call-1", "first", "{}")])]
    )
    harness = _harness(
        gateway,
        executor=SingleLegacyExecutor([Tool("first", "first", "First.", {}, lambda _: "")]),
        event_handler=AcknowledgedSink(),
    )

    async def scenario():
        running = asyncio.create_task(harness.arun(_task()))
        await asyncio.to_thread(sink_waiting.wait, 1)
        running.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await running
        finally:
            acknowledge.set()
        await asyncio.to_thread(executor_finished.wait, 1)

    asyncio.run(scenario())

    assert sink_timeouts == []
    assert executed == []


def test_cancelled_legacy_batch_cannot_start_after_callback_entry_race():
    precheck_finished = threading.Event()
    release_callback = threading.Event()
    executor_finished = threading.Event()
    executed = []

    class PrecheckedLegacyExecutor(Executor):
        def execute_tool_calls(
            self,
            tool_calls,
            *,
            before_call=None,
            after_call=None,
            should_cancel=None,
        ):
            try:
                assert should_cancel is not None
                assert should_cancel() is False
                precheck_finished.set()
                release_callback.wait(1)
                call = tool_calls[0]
                before_call(call)
                executed.append(call.id)
                result = ToolExecutionResult(
                    tool_id=call.id,
                    tool_name=call.name,
                    state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
                )
                after_call(call, result)
                return ToolCallResult(completed_tools=[result], ordered_tools=[result])
            finally:
                executor_finished.set()

    gateway = NativeAsyncGateway(
        [_model(FinishReason.TOOL_CALLS, calls=[LlmToolCall("call-1", "first", "{}")])]
    )
    harness = _harness(
        gateway,
        executor=PrecheckedLegacyExecutor([Tool("first", "first", "First.", {}, lambda _: "")]),
    )

    async def scenario():
        task = _task()
        running = asyncio.create_task(harness.arun(task))
        await asyncio.to_thread(precheck_finished.wait, 1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert task.state is TaskState.TASK_FAILED
        release_callback.set()
        await asyncio.to_thread(executor_finished.wait, 1)

    asyncio.run(scenario())

    assert executed == []


def test_resume_honors_a_legacy_held_call_override():
    called = []

    class LegacyHeldExecutor(DefaultToolExecutor):
        def execute_held_tool_call(self, held):
            return ToolExecutionResult(
                tool_id=held.tool_id,
                tool_name=held.tool_name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                output="legacy denied",
                fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED,
            )

    registry = DefaultToolRegistry()
    registry.register(Tool("shell", "shell", "Shell.", {}, lambda inputs: called.append(inputs)))
    harness = _harness(
        ScriptedGateway([_shell_call(), _model(FinishReason.STOP, "done")]),
        executor=LegacyHeldExecutor(registry),
    )
    task = _task()
    harness.run(task)
    task.decide_tool_call("call_1", approved=True)

    harness.run(task)

    assert called == []
    assert [message.content for message in _tool_messages(task)] == ["legacy denied"]


def test_resumed_tool_executor_error_emits_a_matching_failure():
    class BrokenHeldExecutor(DefaultToolExecutor):
        def execute_held_tool_call(self, held):
            raise RuntimeError("resume crashed")

    registry = DefaultToolRegistry()
    registry.register(Tool("shell", "shell", "Shell.", {}, lambda inputs: "unused"))
    sink = RecordingOutput()
    harness = _harness(
        ScriptedGateway([_shell_call()]),
        executor=BrokenHeldExecutor(registry),
        event_handler=sink,
    )
    task = _task()
    harness.run(task)
    task.decide_tool_call("call_1", approved=True)

    result = harness.run(task)

    assert result.output == "An unexpected error occurred: resume crashed"
    tool_events = [event for event in sink.events if event.type.name.startswith("TOOL_")]
    assert [event.type for event in tool_events[-2:]] == [
        EventType.TOOL_EXECUTION_RESUMED,
        EventType.TOOL_EXECUTION_FAILED,
    ]
    assert tool_events[-1].payload.reason == "unknown"


def test_caller_cancellation_during_resume_keeps_completed_and_settles_remaining_calls():
    async def scenario():
        first_completed = asyncio.Event()
        second_started = asyncio.Event()

        async def first(_inputs):
            first_completed.set()
            return "first completed"

        async def second(_inputs):
            second_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        registry = DefaultToolRegistry()
        registry.register(
            Tool(
                "first",
                "first",
                "First.",
                {},
                first,
                requires_approval=True,
            )
        )
        registry.register(
            Tool(
                "second",
                "second",
                "Second.",
                {},
                second,
                requires_approval=True,
            )
        )
        harness = _harness(
            NativeAsyncGateway(
                [
                    _model(
                        FinishReason.TOOL_CALLS,
                        calls=[
                            LlmToolCall("call-1", "first", "{}"),
                            LlmToolCall("call-2", "second", "{}"),
                        ],
                    )
                ]
            ),
            executor=DefaultToolExecutor(registry),
        )
        task = _task()
        await harness.arun(task)
        task.decide_tool_call("call-1", approved=True)
        task.decide_tool_call("call-2", approved=True)

        resumed = asyncio.create_task(harness.arun(task))
        await first_completed.wait()
        await second_started.wait()
        resumed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resumed
        return task

    task = asyncio.run(scenario())

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL
    assert task.pending_tool_calls == []
    assert [message.content for message in _tool_messages(task)] == [
        "first completed",
        "Tool call cancelled.",
    ]
