"""Run a task by planning once, then alternating model calls and tool calls."""

import time
import uuid

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
from src.schema.event import Event, EventType, LlmEventPayload, TaskEventPayload, ToolEventPayload
from src.schema.harness import StepResult
from src.schema.hook import HarnessHook
from src.schema.llm import (
    FinishReason,
    LLMOutput,
    LlmToolCall,
    assemble_llm_stream,
    is_transient_llm_error,
)
from src.schema.skills import Skill
from src.schema.task import TaskFailReason, TaskPauseReason, TaskPlan, TaskResult, TaskRunningReason
from src.schema.tool import Tool, ToolCallResult, ToolExecutionResult

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
        event_handler: EventHandler | None = None,
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
        self.event_handler = event_handler or EventHandler()
        self._max_steps = max_steps
        self._max_seconds = max_seconds
        self._length_continuations = 0

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

    def _emit_tool_event(
        self,
        event_type: EventType,
        task: Task,
        call: LlmToolCall,
        result=None,
    ) -> None:
        reason = None
        content = None
        name = call.name
        tool_call_id = call.id
        if result is not None:
            name = result.tool_name or call.name
            tool_call_id = result.tool_id or call.id
            reason = result.fail_reason.value if result.fail_reason else None
            content = None if result.output is None else str(result.output)
        self.event_handler.emit(
            Event(
                id=str(uuid.uuid4()),
                type=event_type,
                payload=ToolEventPayload(
                    tool_call_id=tool_call_id,
                    name=name,
                    arguments=call.arguments or None,
                    reason=reason,
                    content=content,
                ),
                task_id=str(task.id),
            )
        )

    def _emit_llm_event(
        self,
        task: Task,
        event_type: EventType,
        *,
        phase: str,
        reason: str | None = None,
        attempt: int | None = None,
        content: str | None = None,
        usage=None,
    ) -> None:
        self.event_handler.emit(
            Event(
                id=str(uuid.uuid4()),
                type=event_type,
                payload=LlmEventPayload(
                    reason=reason,
                    usage=usage,
                    attempt=attempt,
                    phase=phase,
                    content=content,
                ),
                task_id=str(task.id),
            )
        )

    def _emit_answer(self, task: Task, output: LLMOutput, parts: list[str]) -> None:
        if output.requests_tool_calls():
            return
        for part in parts:
            if part:
                self._emit_llm_event(
                    task,
                    EventType.LLM_DELTA,
                    phase="infer",
                    content=part,
                )

    def guard_input(self, task: Task) -> None:
        """
        Reject input that tries to override the harness or disable its controls.
        Args:
            task: The task whose original input is checked.
        Raises:
            GuardError: The input tries to override the harness.
        """
        if self.harness_guard is not None:
            self.harness_guard.guard_input(task)

    def build_planning_context(self, task: Task) -> Context:
        """
        Build the context for the one planning call.
        Args:
            task: The task to plan.
        Returns:
            The planning context.
        Raises:
            ContextBuilderError: The builder failed.
        """
        try:
            return self.context_builder.build_planning_context(
                self.system_prompt, task, tools=self.tools
            )
        except ContextBuilderError:
            raise
        except Exception as error:
            raise ContextBuilderError(str(error)) from error

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
            self._emit_llm_event(task, EventType.LLM_STARTED, phase="planning")
            try:
                output = self.llm_gateway.planning(context, stream=False, response_format=TaskPlan)
                if not isinstance(output, LLMOutput):
                    raise LlmCallError("planning did not return a complete model output")

                if not isinstance(output.structured, TaskPlan):
                    raise LlmCallError("planning did not return a task plan")

                self._emit_llm_event(
                    task,
                    EventType.LLM_COMPLETED,
                    phase="planning",
                    usage=output.usage,
                )
                return output.structured

            except LlmCallError:
                raise

            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                self._emit_llm_event(
                    task,
                    EventType.LLM_RETRYING,
                    phase="planning",
                    reason=type(error).__name__,
                    attempt=attempt,
                )
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)

    def build_infer_context(self, task: Task) -> Context:
        """
        Build the context for one execution step.
        Args:
            task: The task being executed. Its history is the prior transcript.
        Returns:
            The inference context.
        Raises:
            ContextBuilderError: The builder failed.
        """
        try:
            return self.context_builder.build_infer_context(
                self.system_prompt, task, self.tools, self.skills
            )
        except ContextBuilderError:
            raise
        except Exception as error:
            raise ContextBuilderError(str(error)) from error

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
            self._emit_llm_event(task, EventType.LLM_STARTED, phase="infer")
            try:
                produced = self.llm_gateway.infer(context, stream=True)
                output = self._read_inference(task, produced)
                break

            except LlmCallError:
                raise

            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                self._emit_llm_event(
                    task,
                    EventType.LLM_RETRYING,
                    phase="infer",
                    reason=type(error).__name__,
                    attempt=attempt,
                )
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)

        if not isinstance(output, LLMOutput):
            raise LlmCallError("infer did not return a complete model output")

        self._emit_llm_event(task, EventType.LLM_COMPLETED, phase="infer", usage=output.usage)
        return output

    def _read_inference(self, task: Task, produced) -> LLMOutput:
        if isinstance(produced, LLMOutput):
            self._emit_answer(task, produced, [produced.content] if produced.content else [])
            return produced
        try:
            output, parts = assemble_llm_stream(produced)
        except TypeError as error:
            raise LlmCallError(str(error)) from error
        self._emit_answer(task, output, parts)
        return output

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

        pending_tools = []
        completed_tools = []
        failed_tools = []

        for call in tool_calls:
            self._emit_tool_event(EventType.TOOL_EXECUTION_STARTED, task, call)

            result: ToolExecutionResult = self.tool_executor.execute_tool_call(call)
            if result.is_completed():
                completed_tools.append(result)
                self._emit_tool_event(EventType.TOOL_EXECUTION_COMPLETED, task, call, result)
            elif result.is_failed():
                failed_tools.append(result)
                self._emit_tool_event(EventType.TOOL_EXECUTION_FAILED, task, call, result)
            else:
                pending_tools.append(result)

        return ToolCallResult(
            pending_tools=pending_tools, completed_tools=completed_tools, failed_tools=failed_tools
        )

    def step(self, task: Task) -> StepResult:
        """
        Build the context, run one inference, and execute any requested tool calls.
        Args:
            task: The task being executed.
        Returns:
            The model output. Tool calls waiting for approval are recorded and
            reported to the next model call.
        """
        context = self.build_infer_context(task)

        self.harness_hooks.before_think(task=task, context=context)
        output: LLMOutput = self.infer(context, task)
        self.harness_hooks.after_think(task=task, output=output)

        task.add_history_messages([build_history_message(output)])

        if not self._requests_tool_calls(output):
            return StepResult(llm_output=output)

        self.harness_hooks.before_execute_tool_calls(task=task, output=output)
        result: ToolCallResult = self.execute_tool_calls(output.tool_calls, task)
        self.harness_hooks.after_execute_tool_calls(task=task, result=result)

        task.add_history_messages(build_tool_messages(result))

        return StepResult(llm_output=output, tool_call_result=result)

    def build_final_context(self, task: Task) -> Context:
        """
        Build one closing model call from the current transcript.
        Args:
            task: The task being closed. Its history is left unchanged.
        Returns:
            The inference messages plus a closing instruction, and no tools.
        Raises:
            ContextBuilderError: The builder failed.
        """
        try:
            return self.context_builder.build_final_context(
                self.system_prompt, task, tools=self.tools, skills=self.skills
            )
        except ContextBuilderError:
            raise
        except Exception as error:
            raise ContextBuilderError(str(error)) from error

    def finalize(self, task: Task) -> str:
        """
        Close a turn that stopped before a normal answer.
        Args:
            context: The final context.
        Returns:
            The final result.
        """
        final_context = self.build_final_context(task)
        output = self.infer(final_context, task)
        return output.content or ""

    def _blocked(self, step_result: StepResult) -> (bool, TaskPauseReason):
        """
        Check if the turn is blocked.
        Args:
            step_result: The one step result of the turn.
        Returns:
            True if the turn is blocked, False otherwise. And the reason of the blocked.
        """
        if step_result.tool_call_result.pending_tools:
            return True, TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL
        return False, None

    def _failed(self, start_time: float, step_count: int) -> (bool, TaskFailReason):
        """
        Check if the turn is failed.
        Args:
            start_time: The time when the turn started.
            step_count: The number of steps in the turn.
        Returns:
            True if the turn is failed, False otherwise. And the reason of the failed.
        """
        if time.monotonic() - start_time > self._max_seconds:
            return True, TaskFailReason.TASK_FAILED_TIMEOUT
        if step_count >= self._max_steps:
            return True, TaskFailReason.TASK_FAILED_MAX_STEPS
        return False, None

    def _completed(self, step_result: StepResult) -> bool:
        """
        Check if the turn is completed.
        Args:
            step_result: The one step result of the turn.
        Returns:
            True if the turn is completed, False otherwise.
        """
        return step_result.llm_output.finish_reason == FinishReason.STOP

    def _handle_blocked_turn(
        self, task: Task, step_result: StepResult, step_count: int, blocked_reason: TaskPauseReason
    ) -> TaskResult:
        """
        Handle a blocked turn.
        Args:
            task: The task being executed.
            step_result: The one step result of the turn.
            step_count: The number of steps in the turn.
            blocked_reason: The reason of the blocked.
        Returns:
            The task result.
        """
        if blocked_reason == TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL:
            task.set_pending_tool_calls(step_result.tool_call_result.pending_tools)
            task.transition_to_paused(blocked_reason)
            self._emit_task_event(
                EventType.TASK_PAUSED,
                task,
                reason=TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL.value,
                step_id=str(step_count),
            )

        final_output = self.finalize(task)
        return TaskResult(output=final_output)

    def _handle_failed_turn(
        self, task: Task, step_count: int, failed_reason: TaskFailReason
    ) -> TaskResult:
        """
        Handle a failed turn.
        Args:
            task: The task being executed.
            step_count: The number of steps in the turn.
            failed_reason: The reason of the failed.
        Returns:
            The task result.
        """
        task.transition_to_failed(failed_reason)
        self._emit_task_event(
            EventType.TASK_FAILED,
            task,
            reason=failed_reason.value,
            step_id=str(step_count),
        )
        final_output = self.finalize(task)
        return TaskResult(output=final_output)

    def _handle_completed_turn(
        self, task: Task, step_result: StepResult, step_count: int
    ) -> TaskResult:
        """
        Handle a completed turn.
        Args:
            task: The task being executed.
            step_result: The one step result of the turn.
            step_count: The number of steps in the turn.
        Returns:
            The task result.
        """
        result = TaskResult(output=step_result.llm_output.content)

        task.set_result(result)
        task.transition_to_completed()
        self._emit_task_event(
            EventType.TASK_COMPLETED, task, content=result.output, step_id=str(step_count)
        )

        return result

    def turn(self, task: Task) -> TaskResult:
        """
        Infer and execute tools until the model answers or a limit is hit.
        Args:
            task: The task being executed.
        Returns:
            The task result. A completed task carries the assistant text. A paused or
            failed task carries the result recorded while closing the turn.
        """
        start_time = time.monotonic()
        step_count = 0
        while True:
            step_result: StepResult = self.step(task)
            step_count += 1

            blocked, blocked_reason = self._blocked(step_result)
            if blocked:
                return self._handle_blocked_turn(task, step_result, step_count, blocked_reason)

            failed, failed_reason = self._failed(start_time, step_count)
            if failed:
                return self._handle_failed_turn(task, step_count, failed_reason)

            if self._completed(step_result):
                return self._handle_completed_turn(task, step_result, step_count)

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

        except Exception as error:
            if not task.is_failed():
                if isinstance(error, GuardError):
                    reason = TaskFailReason.TASK_FAILED_GUARD_ERROR
                elif isinstance(error, ContextBuilderError):
                    reason = TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR
                elif isinstance(error, LlmCallError):
                    reason = TaskFailReason.TASK_FAILED_LLM_ERROR
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
            return TaskResult(output=f"An unexpected error occurred: {str(error)}")
