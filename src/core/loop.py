"""Run a task by planning once, then alternating model calls and tool calls."""

import time

from src.core.context import (
    ContextBuilder,
    ContextBuilderError,
    build_history_message,
    build_tool_messages,
)
from src.core.guard import GuardError, HarnessGuard
from src.core.llm import LlmCallError, LlmGateway
from src.core.task import Task, get_task_status
from src.core.tool import ToolExecutor
from src.llm.errors import is_transient_llm_error
from src.logger import get_logger, log_context
from src.schema.context import Context
from src.schema.harness import HarnessState
from src.schema.hook import HarnessHook
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall
from src.schema.skills import Skill
from src.schema.task import TaskFailReason, TaskPlan, TaskResult, TaskRunningReason
from src.schema.tool import Tool, ToolCallResult

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_SECONDS = 60 * 30
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_LLM_ATTEMPTS = 5
DEFAULT_LLM_ATTEMPT_DELAY_SECONDS = 10

logger = get_logger("src.core.loop")


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
        harness_hooks: HarnessHook = None,
        harness_guard: HarnessGuard = None,
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
        self._max_steps = max_steps
        self._max_seconds = max_seconds
        self.state = HarnessState.HARNESS_STATE_STARTING

    def transition_to_state(self, state: HarnessState) -> None:
        """
        Transition to a new state.
        Args:
            state: The new state.
        """
        self.state = state

    def guard(self, task: Task) -> None:
        """
        Reject input that tries to override the harness or disable its controls.
        Args:
            task: The task whose original input is checked.
        Raises:
            GuardError: The input tries to override the harness. The task is already failed.
        """
        try:
            self.transition_to_state(HarnessState.HARNESS_STATE_GUARDING)
            if self.harness_guard is not None:
                self.harness_guard.guard_input(task)
        except GuardError:
            if not task.is_failed():
                task.transition_to_failed(TaskFailReason.TASK_FAILED_GUARD_ERROR)
            logger.error("input rejected", extra={"context": get_task_status(task)})
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
            return self.context_builder.build_planning_context(self.system_prompt, task)
        except Exception:
            if not task.is_failed():
                task.transition_to_failed(TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR)
            raise

    def planning(self, context: Context) -> TaskPlan:
        """
        Ask the model for one structured task plan.
        Args:
            context: The planning context.
        Returns:
            The task plan.
        Raises:
            LlmCallError: The model call failed with a permanent error, the plan is missing,
                or the retries ran out.
        """
        self.transition_to_state(HarnessState.HARNESS_STATE_PLANNING)
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
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                logger.warning(
                    "retrying planning call",
                    extra={"context": {"attempt": attempt, "error": type(error).__name__}},
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

    def infer(self, context: Context) -> LLMOutput:
        """
        Run one model call for the execution loop.
        Args:
            context: The inference context.
        Returns:
            The finished model output.
        Raises:
            LlmCallError: The model call failed with a permanent error, or the retries ran out.
        """
        self.transition_to_state(HarnessState.HARNESS_STATE_INFERING)
        for attempt in range(1, DEFAULT_MAX_LLM_ATTEMPTS + 1):
            try:
                output: LLMOutput = self.llm_gateway.infer(context, stream=False)
                break

            except LlmCallError:
                raise

            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                logger.warning(
                    "retrying model call",
                    extra={"context": {"attempt": attempt, "error": type(error).__name__}},
                )
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)

        if not isinstance(output, LLMOutput):
            raise LlmCallError("infer did not return a complete model output")

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

    def execute_tool_calls(self, tool_calls: list[LlmToolCall]) -> ToolCallResult:
        """
        Execute tool calls.
        Args:
            tool_calls: Tool calls from the model, in order.
        Returns:
            Completed, failed, and pending tool executions.
        """
        self.transition_to_state(HarnessState.HARNESS_STATE_TOOL_CALLING)

        return self.tool_executor.execute_tool_calls(list(tool_calls))

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

        if self.harness_hooks is not None and self.harness_hooks.before_think is not None:
            self.harness_hooks.before_think(task=task, context=context)
        output: LLMOutput = self.infer(context)
        if self.harness_hooks is not None and self.harness_hooks.after_think is not None:
            self.harness_hooks.after_think(task=task, output=output)

        task.add_history_message(build_history_message(output))
        if not self._requests_tool_calls(output):
            return output

        if (
            self.harness_hooks is not None
            and self.harness_hooks.before_execute_tool_calls is not None
        ):
            self.harness_hooks.before_execute_tool_calls(task=task, output=output)
        result: ToolCallResult = self.execute_tool_calls(output.tool_calls)
        messages = build_tool_messages(result)
        if (
            self.harness_hooks is not None
            and self.harness_hooks.after_execute_tool_calls is not None
        ):
            self.harness_hooks.after_execute_tool_calls(task=task, result=result, messages=messages)

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
                logger.warning("task timed out", extra={"context": get_task_status(task)})
                return TaskResult()

            if step_count >= self._max_steps:
                task.transition_to_failed(TaskFailReason.TASK_FAILED_MAX_STEPS)
                logger.warning("step limit reached", extra={"context": get_task_status(task)})
                return TaskResult()

            output: LLMOutput = self.step(task)

            if task.is_paused():
                logger.info("task paused", extra={"context": get_task_status(task)})
                return TaskResult()

            if output.finish_reason == FinishReason.STOP:
                result = TaskResult(output.content or "")
                task.set_result(result)
                task.transition_to_completed()
                logger.info("task completed", extra={"context": get_task_status(task)})
                return result

            if self._requests_tool_calls(output):
                step_count += 1
                continue

            task.transition_to_failed(TaskFailReason.TASK_FAILED_LLM_ERROR)
            logger.warning(
                "model stopped without an answer", extra={"context": get_task_status(task)}
            )
            return TaskResult()

    def run(self, task: Task) -> TaskResult:
        """
        Plan a new task, then infer and execute tools until the model answers or a limit is hit.
        Args:
            task: The task to run.
        Returns:
            The task result. Its output is the assistant text when the task is completed.
        """
        with log_context(task_id=str(task.id)):
            try:
                if task.is_finished():
                    logger.info("task already finished", extra={"context": get_task_status(task)})
                    self.transition_to_state(HarnessState.HARNESS_STATE_FINISHED)
                    if isinstance(task.result, TaskResult):
                        return task.result
                    return TaskResult()

                logger.info("task started", extra={"context": get_task_status(task)})
                self.transition_to_state(HarnessState.HARNESS_STATE_RUNNING)

                if task.is_created():
                    self.guard(task)
                    logger.info("planning started")

                    task.set_plan(self.planning(self.build_planning_context(task)))
                    logger.info("plan ready", extra={"context": {"goal": task.plan.goal}})

                    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
                    return TaskResult()

                if task.is_paused():
                    logger.info("task resumed")
                    task.transition_to_running(TaskRunningReason.TASK_RUNNING_RESUME)

                if task.is_queued():
                    logger.info("queued task started")
                    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)

                result = self.turn(task)
                self.transition_to_state(HarnessState.HARNESS_STATE_FINISHED)
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

                logger.exception("task failed", extra={"context": get_task_status(task)})
                raise
