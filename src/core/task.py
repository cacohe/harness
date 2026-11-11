from __future__ import annotations

import uuid
from typing import FrozenSet, Optional

from pydantic import Field

from src.schema.context import Message
from src.schema.task import (
    TaskFailReason,
    TaskPauseReason,
    TaskPlan,
    TaskResult,
    TaskRunningReason,
    TaskState,
    TaskStateReason,
)
from src.schema.tool import ToolExecutionResult


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
        id: str = uuid.uuid4(),
        state: TaskState = TaskState.TASK_CREATED,
        state_reason: TaskStateReason = None,
        plan: TaskPlan = None,
        history: list[Message] = [],
        pending_tool_calls: list[ToolExecutionResult] = [],
        result: TaskResult = None,
    ):
        self.id = id
        self.original_input = original_input
        self.state = state
        self.state_reason = state_reason
        self.plan = plan
        self.history = history
        self.pending_tool_calls = pending_tool_calls
        self.result = result

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
            "plan": self.plan.to_dict(),
            "history": self.history,
            "pending_tool_calls": self.pending_tool_calls,
            "result": self.result.to_dict() if self.result else None,
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

        self.persist()

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
        self.persist()

    def add_history_messages(self, messages: list[Message]) -> None:
        """
        Add a message to the task history.
        Args:
            message: The message to add to the task history.
        """
        self.history.extend(messages)
        self.persist()

    def set_pending_tool_calls(self, pending_tool_calls: list[ToolExecutionResult]) -> None:
        """
        Replace the tool calls waiting to run.
        Args:
            pending_tool_calls: Tool executions that have not finished.
        """
        self.pending_tool_calls = pending_tool_calls
        self.persist()

    def set_result(self, result: TaskResult) -> None:
        """
        Replace the task result.
        Args:
            result: The result produced for this task.
        """
        self.result = result
        self.persist()

    def persist(self) -> None:
        """
        Persist the task.
        """
        ...


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
