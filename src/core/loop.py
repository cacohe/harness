"""Run a task by planning once, then alternating model calls and tool calls."""

import json
import time
import uuid

from pydantic import ValidationError

from src.core.context import (
    ContextBuilder,
    ContextBuilderError,
    build_history_message,
    build_tool_messages,
)
from src.core.event import EventHandler
from src.core.guard import GuardError, HarnessGuard
from src.core.llm import LlmCallError, LlmGateway
from src.core.task import Task
from src.core.tool import ToolExecutor
from src.schema.context import Context
from src.schema.event import Event, EventType, LlmEventPayload, TaskEventPayload
from src.schema.hook import HarnessHook
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall, is_transient_llm_error
from src.schema.skills import Skill
from src.schema.task import TaskFailReason, TaskPlan, TaskResult, TaskRunningReason
from src.schema.tool import Tool, ToolCallResult

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_SECONDS = 60 * 30
DEFAULT_MAX_LLM_ATTEMPTS = 5
DEFAULT_LLM_ATTEMPT_DELAY_SECONDS = 10
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class Harness:
    """Run one task from planning through tool use until the model answers or a limit is hit."""

    def __init__(
        self,
        system_prompt: str,
        llm_gateway: LlmGateway,
        tools: list[Tool],
        skills: list[Skill],
        tool_executor: ToolExecutor,
        context_builder: ContextBuilder,
        harness_hooks: HarnessHook = HarnessHook(),
        harness_guard: HarnessGuard = HarnessGuard(),
        event_handler: EventHandler = EventHandler(),
        max_steps: int = DEFAULT_MAX_STEPS,
        max_seconds: int = DEFAULT_MAX_SECONDS,
    ):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.llm_gateway = llm_gateway
        self.tools = tools
        self.skills = skills
        self.tool_executor = tool_executor
        self.context_builder = context_builder
        self.harness_hooks = harness_hooks
        self.harness_guard = harness_guard
        self.event_handler = event_handler
        self.tool_executor.event_handler = self.event_handler
        self._max_steps = max_steps
        self._max_seconds = max_seconds

    def _emit_task_event(
        self,
        event_type: EventType,
        task: Task,
        *,
        reason: str | None = None,
        content: str | None = None,
        goal: list[str] | None = None,
        step_id: str | None = None,
    ) -> None:
        task_id = str(task.id)
        self.event_handler.emit(
            Event(
                id=str(uuid.uuid4()),
                type=event_type,
                payload=TaskEventPayload(
                    task_id=task_id,
                    reason=reason,
                    content=content,
                    goal=goal,
                ),
                task_id=task_id,
                step_id=step_id,
            )
        )

    def _emit_llm_retry_event(self, task: Task, phase: str, attempt: int, error: Exception) -> None:
        event_id = str(uuid.uuid4())
        self.event_handler.emit(
            Event(
                id=event_id,
                type=EventType.LLM_RETRYING,
                payload=LlmEventPayload(
                    llm_id=event_id,
                    reason=type(error).__name__,
                    attempt=attempt,
                    phase=phase,
                ),
                task_id=str(task.id),
            )
        )

    def guard_input(self, task: Task) -> None:
        """
        Reject input that tries to override the harness or disable its controls.
        Args:
            task: The task whose original input is checked.
        Raises:
            GuardError: The input tries to override the harness. The task is already failed.
        """
        try:
            if self.harness_guard is not None:
                self.harness_guard.guard_input(task)
        except GuardError:
            if not task.is_failed():
                task.transition_to_failed(TaskFailReason.TASK_FAILED_GUARD_ERROR)
            self._emit_task_event(
                EventType.TASK_FAILED,
                task,
                reason=TaskFailReason.TASK_FAILED_GUARD_ERROR.value,
                content="input rejected",
            )
            raise

    def build_planning_context(self, task: Task) -> Context:
        """
        Build the context for the one planning call.
        Args:
            task: The task to plan.
        Returns:
            The planning context.
        Raises:
            ContextBuilderError: The builder failed. The task is already failed.
        """
        try:
            return self.context_builder.build_planning_context(
                self.system_prompt, task, tools=self.tools
            )
        except Exception:
            if not task.is_failed():
                task.transition_to_failed(TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR)
            raise

    def planning(self, context: Context, task: Task) -> TaskPlan:
        """
        Ask the model for one structured task plan.
        Args:
            context: The planning context.
            task: The task being planned. Retry events are emitted for this task.
        Returns:
            The task plan.
        Raises:
            LlmCallError: The model call failed with a permanent error, the plan is missing,
                or the retries ran out. Invalid structured output is retried.
        """
        for attempt in range(1, DEFAULT_MAX_LLM_ATTEMPTS + 1):
            try:
                output = self.llm_gateway.planning(context, stream=False, response_format=TaskPlan)
                if not isinstance(output, LLMOutput):
                    raise LlmCallError("planning did not return a complete model output")

                if not isinstance(output.structured, TaskPlan):
                    raise LlmCallError("planning did not return a task plan")

                return output.structured

            except LlmCallError:
                raise

            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not _can_retry_planning(error):
                    raise LlmCallError(str(error)) from error
                self._emit_llm_retry_event(task, "planning", attempt, error)
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)

    def build_infer_context(self, task: Task) -> Context:
        """
        Build the context for one execution step.
        Args:
            task: The task being executed. Its history is the prior transcript.
        Returns:
            The inference context.
        Raises:
            ContextBuilderError: The builder failed. The task is already failed.
        """
        try:
            return self.context_builder.build_infer_context(
                self.system_prompt, task, self.tools, self.skills
            )
        except Exception:
            if not task.is_failed():
                task.transition_to_failed(TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR)
            raise

    def infer(self, context: Context, task: Task) -> LLMOutput:
        """
        Run one model call for the execution loop.
        Args:
            context: The inference context.
            task: The task being executed. Text deltas and retry events are emitted for this task.
        Returns:
            The finished model output. Text is streamed by default and assembled into this result.
        Raises:
            LlmCallError: The model call failed with a permanent error, or the retries ran out.
        """
        for attempt in range(1, DEFAULT_MAX_LLM_ATTEMPTS + 1):
            try:
                produced = self.llm_gateway.infer(context, stream=True)
                output = self._read_inference(task, produced)
                break

            except LlmCallError:
                raise

            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                self._emit_llm_retry_event(task, "infer", attempt, error)
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)

        if not isinstance(output, LLMOutput):
            raise LlmCallError("infer did not return a complete model output")

        return output

    def _read_inference(self, task: Task, produced) -> LLMOutput:
        if isinstance(produced, LLMOutput):
            self._emit_answer(task, produced, [produced.content] if produced.content else [])
            return produced
        return self._consume_llm_stream(task, produced)

    def _consume_llm_stream(self, task: Task, pieces) -> LLMOutput:
        text: list[str] = []
        calls: list[LlmToolCall] = []
        finish_reason = None
        usage = None
        reasoning = None
        thinking = None
        structured = None
        for piece in pieces:
            if not isinstance(piece, LLMOutput):
                raise LlmCallError("infer did not return a model output")
            if piece.content:
                text.append(piece.content)
            _merge_tool_calls(calls, piece.tool_calls)
            if piece.finish_reason is not None:
                finish_reason = piece.finish_reason
            if piece.usage is not None:
                usage = piece.usage
            if piece.reasoning_content:
                reasoning = piece.reasoning_content
            if piece.thinking_blocks:
                thinking = piece.thinking_blocks
            if piece.structured is not None:
                structured = piece.structured
        output = LLMOutput(
            finish_reason,
            "".join(text) or None,
            calls,
            reasoning,
            thinking,
            usage,
        )
        output.structured = structured
        self._emit_answer(task, output, text)
        return output

    def _emit_answer(self, task: Task, output: LLMOutput, parts: list[str]) -> None:
        if output.requests_tool_calls():
            return
        for part in parts:
            if part:
                self._emit_llm_delta(task, part)

    def _emit_llm_delta(self, task: Task, content: str) -> None:
        event_id = str(uuid.uuid4())
        self.event_handler.emit(
            Event(
                id=event_id,
                type=EventType.LLM_DELTA,
                payload=LlmEventPayload(llm_id=event_id, content=content, phase="infer"),
                task_id=str(task.id),
            )
        )

    def _requests_tool_calls(self, llm_output: LLMOutput) -> bool:
        """
        Check if the model output requests tool calls.
        Args:
            llm_output: The model output.
        Returns:
            True if the model output requests tool calls, False otherwise.
        """
        return llm_output.requests_tool_calls()

    def execute_tool_calls(self, tool_calls: list[LlmToolCall], task: Task) -> ToolCallResult:
        """
        Execute tool calls.
        Args:
            tool_calls: Tool calls from the model, in order.
            task: The task whose id is attached to the tool events.
        Returns:
            Completed, failed, and pending tool executions.
        """

        return self.tool_executor.execute_tool_calls(list(tool_calls), task_id=str(task.id))

    def step(self, task: Task) -> LLMOutput:
        """
        Build the context, run one inference, and execute any requested tool calls.
        Args:
            task: The task being executed.
        Returns:
            The model output. Tool calls waiting for approval are recorded and
            reported to the next model call.
        """
        context = self.build_infer_context(task)

        if self.harness_hooks.before_think:
            self.harness_hooks.before_think(task=task, context=context)
        output: LLMOutput = self.infer(context, task)
        if self.harness_hooks.after_think:
            self.harness_hooks.after_think(task=task, output=output)

        task.add_history_message(build_history_message(output))
        if not self._requests_tool_calls(output):
            return output

        if self.harness_hooks.before_execute_tool_calls:
            self.harness_hooks.before_execute_tool_calls(task=task, output=output)
        result: ToolCallResult = self.execute_tool_calls(output.tool_calls, task)
        if self.harness_hooks.after_execute_tool_calls:
            self.harness_hooks.after_execute_tool_calls(task=task, result=result)

        messages = build_tool_messages(result)
        for message in messages:
            task.add_history_message(message)

        if result.pending_tools:
            task.set_pending_tool_calls(result.pending_tools)

        return output

    def turn(self, task: Task) -> TaskResult:
        started = time.monotonic()
        step_count = 0
        while True:
            if time.monotonic() - started > self._max_seconds:
                task.transition_to_failed(TaskFailReason.TASK_FAILED_TIMEOUT)
                self._emit_task_event(
                    EventType.TASK_FAILED,
                    task,
                    reason=TaskFailReason.TASK_FAILED_TIMEOUT.value,
                    content="task timed out",
                    step_id=str(step_count),
                )
                return task.result

            if step_count >= self._max_steps:
                task.transition_to_failed(TaskFailReason.TASK_FAILED_MAX_STEPS)
                self._emit_task_event(
                    EventType.TASK_FAILED,
                    task,
                    reason=TaskFailReason.TASK_FAILED_MAX_STEPS.value,
                    content="step limit reached",
                    step_id=str(step_count),
                )
                return task.result

            output: LLMOutput = self.step(task)

            if task.is_paused():
                reason = task.state_reason.value if task.state_reason else None
                self._emit_task_event(
                    EventType.TASK_PAUSED,
                    task,
                    reason=reason,
                    step_id=str(step_count),
                )
                return TaskResult()

            if output.finish_reason == FinishReason.STOP:
                result = TaskResult(output.content or "")
                task.set_result(result)
                task.transition_to_completed()
                self._emit_task_event(
                    EventType.TASK_COMPLETED,
                    task,
                    content=result.output,
                    step_id=str(step_count),
                )
                return result

            if self._requests_tool_calls(output):
                step_count += 1
                continue

            task.transition_to_failed(TaskFailReason.TASK_FAILED_LLM_ERROR)
            self._emit_task_event(
                EventType.TASK_FAILED,
                task,
                reason=TaskFailReason.TASK_FAILED_LLM_ERROR.value,
                content="model stopped without an answer",
                step_id=str(step_count),
            )
            return task.result

    def run(self, task: Task) -> TaskResult:
        """
        Plan a new task, then infer and execute tools until the model answers or a limit is hit.
        Args:
            task: The task to run.
        Returns:
            The task result. Its output is the assistant text when the task is completed.
        """
        try:
            if task.is_finished():
                self._emit_task_event(EventType.TASK_FINISHED, task)
                return task.result

            self._emit_task_event(EventType.TASK_STARTED, task)

            if task.is_created():
                task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
                self.guard_input(task)
                self._emit_task_event(EventType.TASK_PLANNING, task)

                task.set_plan(self.planning(self.build_planning_context(task), task))
                self._emit_task_event(EventType.TASK_PLANNED, task, goal=list(task.plan.goal))

            if task.is_paused():
                self._emit_task_event(EventType.TASK_RESUMED, task)
                task.transition_to_running(TaskRunningReason.TASK_RUNNING_RESUME)

            result = self.turn(task)
            return result

        except GuardError:
            raise

        except Exception as error:
            if not task.is_failed():
                if isinstance(error, LlmCallError):
                    reason = TaskFailReason.TASK_FAILED_LLM_ERROR
                elif isinstance(error, ContextBuilderError):
                    reason = TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR
                else:
                    reason = TaskFailReason.TASK_FAILED_UNKNOWN
                task.transition_to_failed(reason)

            fail_reason = task.state_reason.value if task.state_reason else None
            self._emit_task_event(
                EventType.TASK_FAILED,
                task,
                reason=fail_reason,
                content=str(error),
            )
            raise


def _can_retry_planning(error: Exception) -> bool:
    return is_transient_llm_error(error) or isinstance(
        error, (ValidationError, json.JSONDecodeError)
    )


def _merge_tool_calls(calls: list[LlmToolCall], deltas: list[LlmToolCall]) -> None:
    for delta in deltas:
        if delta.id or not calls:
            calls.append(LlmToolCall(delta.id, delta.name, delta.arguments))
            continue
        current = calls[-1]
        if delta.name and not current.name:
            current.name = delta.name
        current.arguments += delta.arguments
