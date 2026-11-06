"""Run a task by planning once, then alternating model calls and tool calls."""

import time

from src.core.context import Context, ContextBuilder, ContextBuilderError, Message
from src.core.guard import GuardError
from src.core.guard import guard as guard_input
from src.core.hook import HarnessHook
from src.core.llm import ROLE_ASSISTANT, FinishReason, LlmCallError, LlmGateway, LLMOutput
from src.core.task import Task, TaskFailReason, TaskPauseReason, TaskRunningReason, TaskState
from src.core.tool import ToolExecutor
from src.llm.errors import is_transient_llm_error
from src.llm.plan import TaskPlanSchema, task_plan_from_structured
from src.logger import get_logger, log_context
from src.tool.tool_calls import collect_tool_results

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_SECONDS = 60 * 30
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_LLM_ATTEMPTS = 5
DEFAULT_LLM_ATTEMPT_DELAY_SECONDS = 10

_TOOL_FINISH_REASONS = {FinishReason.TOOL_CALLS, FinishReason.FUNCTION_CALL}
_log = get_logger("core.loop")


def _status(task: Task) -> dict:
    fields = {"state": task.state.value}
    if task.state in {TaskState.TASK_FAILED, TaskState.TASK_PAUSED} and task.state_reason:
        fields["reason"] = task.state_reason.value
    return fields


def _finish_name(finish_reason: FinishReason | None) -> str | None:
    if finish_reason is None:
        return None
    return finish_reason.value


class Harness:
    """Run one task from planning through tool use until the model answers or a limit is hit."""

    def __init__(
        self,
        llm_gateway: LlmGateway,
        tool_executor: ToolExecutor,
        context_builder: ContextBuilder,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_seconds: int = DEFAULT_MAX_SECONDS,
        hooks: HarnessHook | None = None,
    ):
        self.llm_gateway = llm_gateway
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.context_builder = context_builder
        self._max_steps = max_steps
        self._max_seconds = max_seconds
        self.hooks = hooks or HarnessHook()

    def guard(self, task: Task) -> None:
        """
        Reject input that tries to override the harness or disable its controls.
        Args:
            task: The task whose original input is checked.
        Raises:
            GuardError: The input tries to override the harness. The task is already failed.
        """
        try:
            guard_input(task)
        except GuardError:
            if task.state != TaskState.TASK_FAILED:
                task.transition_to(TaskState.TASK_FAILED, TaskFailReason.TASK_FAILED_GUARD_ERROR)
            _log.error("input rejected", extra={"context": _status(task)})
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
            if task.state != TaskState.TASK_FAILED:
                task.transition_to(
                    TaskState.TASK_FAILED, TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR
                )
            raise

    def planning(self, context: Context) -> LLMOutput:
        """
        Ask the model for one structured task plan.
        Args:
            context: The planning context.
        Returns:
            The model output. structured is a TaskPlanSchema when planning succeeds.
        Raises:
            LlmCallError: The model call failed with a permanent error, or the retries ran out.
        """
        for attempt in range(1, DEFAULT_MAX_LLM_ATTEMPTS + 1):
            try:
                return self.llm_gateway.planning(
                    context, stream=False, response_format=TaskPlanSchema
                )
            except LlmCallError:
                raise
            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                _log.warning(
                    "retrying planning call",
                    extra={"context": {"attempt": attempt, "error": type(error).__name__}},
                )
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)

    def _prepare(self, task: Task) -> None:
        if task.state == TaskState.TASK_CREATED:
            self.guard(task)
            _log.info("planning started")
            output = self.planning(self.build_planning_context(task))
            task.set_plan(task_plan_from_structured(output.structured))
            _log.info("plan ready", extra={"context": {"goal": task.plan.goal}})
            task.transition_to(TaskState.TASK_RUNNING, TaskRunningReason.TASK_RUNNING_CREATE)
            return

        if task.state == TaskState.TASK_PAUSED:
            _log.info("task resumed")
            task.transition_to(TaskState.TASK_RUNNING, TaskRunningReason.TASK_RUNNING_RESUME)
            return

        if task.state == TaskState.TASK_QUEUE:
            _log.info("queued task started")
            task.transition_to(TaskState.TASK_RUNNING, TaskRunningReason.TASK_RUNNING_CREATE)

    def build_infer_context(self, task: Task, history: list | None = None) -> Context:
        """
        Build the context for one execution step.
        Args:
            task: The task being executed.
            history: Assistant and tool messages from earlier steps.
        Returns:
            The inference context.
        Raises:
            ContextBuilderError: The builder failed. The task is already failed.
        """
        try:
            return self.context_builder.build_infer_context(
                self.system_prompt, task, history=history or []
            )
        except Exception:
            if task.state != TaskState.TASK_FAILED:
                task.transition_to(
                    TaskState.TASK_FAILED, TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR
                )
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
        for attempt in range(1, DEFAULT_MAX_LLM_ATTEMPTS + 1):
            try:
                output = self.llm_gateway.infer(context, stream=False)
                break
            except LlmCallError:
                raise
            except Exception as error:
                if attempt == DEFAULT_MAX_LLM_ATTEMPTS or not is_transient_llm_error(error):
                    raise LlmCallError(str(error)) from error
                _log.warning(
                    "retrying model call",
                    extra={"context": {"attempt": attempt, "error": type(error).__name__}},
                )
                time.sleep(DEFAULT_LLM_ATTEMPT_DELAY_SECONDS)
        if not isinstance(output, LLMOutput):
            raise LlmCallError("infer did not return a complete model output")
        return output

    def execute_tool_calls(self, task: Task, calls: list) -> list[Message] | None:
        """
        Execute tool calls until one needs approval.
        Args:
            task: The task that owns the pending calls and the transcript.
            calls: Tool calls to run, in order.
        Returns:
            One tool message per completed call. None when a call is waiting for approval.
        """
        batch = collect_tool_results(self.tool_executor, list(calls))
        for message in batch.messages:
            if (message.content or "").startswith("Error:"):
                _log.warning("tool failed", extra={"context": {"tool": message.name}})
            else:
                _log.info("tool finished", extra={"context": {"tool": message.name}})
        task.history.extend(batch.messages)
        task.pending_tool_calls = batch.pending
        if not batch.waiting_for_approval:
            return batch.messages
        waiting = batch.pending[0].name if batch.pending else None
        _log.info("tool waiting for approval", extra={"context": {"tool": waiting}})
        task.transition_to(TaskState.TASK_PAUSED, TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL)
        return None

    def step(self, task: Task, context: Context) -> LLMOutput:
        """
        Run one inference and, when requested, its tool calls.
        Args:
            task: The task being executed.
            context: The context for this step.
        Returns:
            The model output. The task is paused when a tool call is waiting for approval.
        """
        if self.hooks.before_think is not None:
            self.hooks.before_think(task, context)
        output = self.infer(context)
        if self.hooks.after_think is not None:
            self.hooks.after_think(task, output)
        _log.info(
            "model responded",
            extra={
                "context": {
                    "finish_reason": _finish_name(output.finish_reason),
                    "tools": [call.name for call in output.tool_calls],
                }
            },
        )

        task.history.append(
            Message(
                role=ROLE_ASSISTANT,
                content=output.content,
                tool_calls=list(output.tool_calls),
                reasoning_content=output.reasoning_content,
                thinking_blocks=output.thinking_blocks,
            )
        )
        if output.finish_reason not in _TOOL_FINISH_REASONS or not output.tool_calls:
            return output

        if self.hooks.before_execute_tool_calls is not None:
            self.hooks.before_execute_tool_calls(task, output)
        messages = self.execute_tool_calls(task, output.tool_calls)
        if messages is None:
            return output
        if self.hooks.after_execute_tool_calls is not None:
            self.hooks.after_execute_tool_calls(task, messages)
        return output

    def turn(self, task: Task) -> str:
        if (
            task.pending_tool_calls
            and self.execute_tool_calls(task, task.pending_tool_calls) is None
        ):
            _log.info("task paused", extra={"context": _status(task)})
            return ""
        started = time.monotonic()
        step_count = 0
        while True:
            if time.monotonic() - started > self._max_seconds:
                if task.state != TaskState.TASK_FAILED:
                    task.transition_to(TaskState.TASK_FAILED, TaskFailReason.TASK_FAILED_TIMEOUT)
                _log.warning("task timed out", extra={"context": _status(task)})
                return ""

            if step_count >= self._max_steps:
                if task.state != TaskState.TASK_FAILED:
                    task.transition_to(TaskState.TASK_FAILED, TaskFailReason.TASK_FAILED_MAX_STEPS)
                _log.warning("step limit reached", extra={"context": _status(task)})
                return ""

            output = self.step(task, self.build_infer_context(task, task.history))

            if task.state == TaskState.TASK_PAUSED:
                _log.info("task paused", extra={"context": _status(task)})
                return ""
            if output.finish_reason == FinishReason.STOP:
                task.result = output.content or ""
                task.transition_to(TaskState.TASK_COMPLETED)
                _log.info("task completed", extra={"context": _status(task)})
                return task.result
            if output.finish_reason in _TOOL_FINISH_REASONS and output.tool_calls:
                step_count += 1
                continue
            if task.state != TaskState.TASK_FAILED:
                task.transition_to(TaskState.TASK_FAILED, TaskFailReason.TASK_FAILED_LLM_ERROR)
            _log.warning("model stopped without an answer", extra={"context": _status(task)})
            return ""

    def run(self, task: Task) -> str:
        """
        Plan a new task, then infer and execute tools until the model answers or a limit is hit.
        Args:
            task: The task to run. A paused task resumes its pending tool calls.
        Returns:
            The final assistant text when the task is completed, otherwise an empty string.
            The task state records failure, completion, and approval pauses.
        """
        with log_context(task_id=str(task.id)):
            try:
                if task.state in {TaskState.TASK_COMPLETED, TaskState.TASK_FAILED}:
                    _log.info("task already finished", extra={"context": _status(task)})
                    return task.result or ""

                _log.info("task started", extra={"context": _status(task)})
                self._prepare(task)
                return self.turn(task)

            except GuardError:
                raise

            except Exception as error:
                if task.state != TaskState.TASK_FAILED:
                    if isinstance(error, LlmCallError):
                        reason = TaskFailReason.TASK_FAILED_LLM_ERROR
                    elif isinstance(error, ContextBuilderError):
                        reason = TaskFailReason.TASK_FAILED_CONTEXT_BUILD_ERROR
                    else:
                        reason = TaskFailReason.TASK_FAILED_UNKNOWN
                    task.transition_to(TaskState.TASK_FAILED, reason)
                _log.exception("task failed", extra={"context": _status(task)})
                raise
