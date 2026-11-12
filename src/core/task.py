from __future__ import annotations

import uuid
from typing import FrozenSet, Optional

from src.schema.context import Message
from src.schema.llm import ROLE_TOOL
from src.schema.task import (
    TaskFailReason,
    TaskPauseReason,
    TaskPlan,
    TaskResult,
    TaskRunningReason,
    TaskState,
    TaskStateReason,
)
from src.schema.tool import ApprovalDecision, ToolExecutionResult


class InvalidTransition(Exception):
    """Invalid transition exception."""

    def __init__(self, current_state: TaskState, new_state: TaskState) -> None:
        super().__init__(f"Invalid transition: {current_state} -> {new_state}")


class Task:
    """A task to be executed by the harness."""

    TRANSITIONS: dict[TaskState, FrozenSet[TaskState]] = {
        TaskState.TASK_CREATED: frozenset([TaskState.TASK_RUNNING, TaskState.TASK_FAILED]),
        TaskState.TASK_RUNNING: frozenset(
            [TaskState.TASK_PAUSED, TaskState.TASK_FAILED, TaskState.TASK_COMPLETED]
        ),
        TaskState.TASK_PAUSED: frozenset([TaskState.TASK_RUNNING]),
        TaskState.TASK_FAILED: frozenset([]),
        TaskState.TASK_COMPLETED: frozenset([]),
    }

    def __init__(
        self,
        original_input: str,
        id: str | None = None,
        state: TaskState = TaskState.TASK_CREATED,
        state_reason: TaskStateReason = None,
        plan: TaskPlan | None = None,
        history: list[Message] | None = None,
        pending_tool_calls: list[ToolExecutionResult] | None = None,
        result: TaskResult | None = None,
    ):
        self.id = id if id is not None else str(uuid.uuid4())
        self.original_input = original_input
        self.state = state
        self.state_reason = state_reason
        self.plan = plan
        self.history = list(history) if history is not None else []
        self.pending_tool_calls = (
            list(pending_tool_calls) if pending_tool_calls is not None else []
        )
        self.result = result
        self.step_count = 0
        self.elapsed_seconds = 0.0

    def to_dict(self) -> dict:
        """
        Convert the task to a dictionary.
        Returns:
            A dictionary representation of the task.
        """
        return {
            "id": self.id,
            "original_input": self.original_input,
            "state": self.state.value,
            "state_reason": self.state_reason.value if self.state_reason else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "history": self.history,
            "pending_tool_calls": self.pending_tool_calls,
            "result": self.result.to_dict() if self.result else None,
            "step_count": self.step_count,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def _transition_to(
        self, state: TaskState, state_reason: Optional[TaskStateReason] = None
    ) -> None:
        """
        Transition the task to a new state.
        Args:
            state: The new state.
            state_reason: The reason for the state transition.
        """
        if state not in self.TRANSITIONS[self.state]:
            raise InvalidTransition(self.state, state)

        self.state = state
        if state_reason is not None:
            self.state_reason = state_reason

    def transition_to_running(self, reason: TaskRunningReason) -> None:
        """
        Run the task.
        """
        self._transition_to(TaskState.TASK_RUNNING, reason)

    def transition_to_paused(self, reason: TaskPauseReason) -> None:
        """
        Pause the task.
        Args:
            reason: The reason for the task pause.
        """
        self._transition_to(TaskState.TASK_PAUSED, reason)

    def transition_to_failed(self, reason: TaskFailReason) -> None:
        """
        Fail the task and keep an empty result when the task has none.
        Args:
            reason: The reason for the task failure.
        """
        if self.result is None:
            self.result = TaskResult()
        self._transition_to(TaskState.TASK_FAILED, reason)

    def transition_to_completed(self) -> None:
        """
        Complete the task.
        """
        self._transition_to(TaskState.TASK_COMPLETED)

    def is_created(self) -> bool:
        """
        Check if the task is created.
        Returns:
            True if the task is created, False otherwise.
        """
        return self.state == TaskState.TASK_CREATED

    def is_running(self) -> bool:
        """
        Check if the task is running.
        Returns:
            True if the task is running, False otherwise.
        """
        return self.state == TaskState.TASK_RUNNING

    def is_paused(self) -> bool:
        """
        Check if the task is paused.
        Returns:
            True if the task is paused, False otherwise.
        """
        return self.state == TaskState.TASK_PAUSED

    def is_failed(self) -> bool:
        """
        Check if the task is failed.
        Returns:
            True if the task is failed, False otherwise.
        """
        return self.state == TaskState.TASK_FAILED

    def is_completed(self) -> bool:
        """
        Check if the task is completed.
        Returns:
            True if the task is completed, False otherwise.
        """
        return self.state == TaskState.TASK_COMPLETED

    def is_finished(self) -> bool:
        """
        Check if the task is finished.
        Returns:
            True if the task is finished, False otherwise.
        """
        return self.is_completed() or self.is_failed()

    def set_plan(self, plan: TaskPlan) -> None:
        """
        Replace the task plan.
        Args:
            plan: The plan produced for this task.
        """
        self.plan = plan

    def add_history_messages(self, messages: list[Message]) -> None:
        """
        Add a message to the task history.
        Args:
            message: The message to add to the task history.
        """
        self.history.extend(messages)

    def set_pending_tool_calls(self, pending_tool_calls: list[ToolExecutionResult]) -> None:
        """
        Replace the tool calls waiting to run.
        Args:
            pending_tool_calls: Tool executions that have not finished.
        """
        self.pending_tool_calls = pending_tool_calls

    def approval_requests(self) -> list[ToolExecutionResult]:
        """
        List the pending tool calls that still wait for the user's decision.
        Returns:
            Undecided calls that need approval, in call order.
        """
        return [
            item
            for item in self.pending_tool_calls
            if item.needs_approval and item.approval is None
        ]

    def decide_tool_call(self, tool_id: str, approved: bool) -> None:
        """
        Record the user's decision on one tool call waiting for approval.
        Args:
            tool_id: The id of the pending tool call.
            approved: True approves the call. False rejects it.
        Raises:
            ValueError: No pending tool call needs approval with that id.
        """
        for item in self.pending_tool_calls:
            if item.tool_id == tool_id and item.needs_approval:
                item.approval = (
                    ApprovalDecision.APPROVAL_DECISION_APPROVED
                    if approved
                    else ApprovalDecision.APPROVAL_DECISION_REJECTED
                )
                return
        raise ValueError(f"no tool call waits for approval with id {tool_id}")

    def replace_tool_messages(self, messages: list[Message]) -> None:
        """
        Replace history tool messages that answer the same tool calls.
        Args:
            messages: Tool messages. A message without a matching tool call is appended.
        """
        by_call_id = {message.tool_call_id: message for message in messages}
        for index, message in enumerate(self.history):
            if message.role == ROLE_TOOL and message.tool_call_id in by_call_id:
                self.history[index] = by_call_id.pop(message.tool_call_id)
        self.history.extend(by_call_id.values())

    def set_progress(self, step_count: int, elapsed_seconds: float) -> None:
        """
        Record how many steps ran and how long the task ran before it paused.
        Args:
            step_count: Steps already taken.
            elapsed_seconds: Running time already used, excluding paused time.
        """
        self.step_count = step_count
        self.elapsed_seconds = elapsed_seconds

    def set_result(self, result: TaskResult) -> None:
        """
        Replace the task result.
        Args:
            result: The result produced for this task.
        """
        self.result = result


def get_task_status(task: Task) -> dict:
    """
    Get the status of the task.
    Args:
        task: The task to get the status of.
    Returns:
        A dictionary containing the status of the task.
    """
    status = {"state": task.state.value}
    if task.state in {TaskState.TASK_FAILED, TaskState.TASK_PAUSED} and task.state_reason:
        status["reason"] = task.state_reason.value
    return status
