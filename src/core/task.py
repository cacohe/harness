from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar, FrozenSet

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

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


class Task(BaseModel):
    """A task to be executed by the harness."""

    model_config = ConfigDict(validate_assignment=False)

    TRANSITIONS: ClassVar[dict[TaskState, FrozenSet[TaskState]]] = {
        TaskState.TASK_CREATED: frozenset([TaskState.TASK_RUNNING, TaskState.TASK_FAILED]),
        TaskState.TASK_RUNNING: frozenset(
            [TaskState.TASK_PAUSED, TaskState.TASK_FAILED, TaskState.TASK_COMPLETED]
        ),
        TaskState.TASK_PAUSED: frozenset([TaskState.TASK_RUNNING, TaskState.TASK_FAILED]),
        TaskState.TASK_FAILED: frozenset([]),
        TaskState.TASK_COMPLETED: frozenset([]),
    }

    schema_version: int = 1
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_input: str
    state: TaskState = TaskState.TASK_CREATED
    state_reason: TaskStateReason | None = None
    plan: TaskPlan | None = None
    history: list[Message] = Field(default_factory=list)
    pending_tool_calls: list[ToolExecutionResult] = Field(default_factory=list)
    result: TaskResult | None = None
    step_count: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def __init__(self, original_input: str, id: str | None = None, **data: Any):
        if id is not None:
            data["id"] = id
        if data.get("history") is None:
            data.pop("history", None)
        if data.get("pending_tool_calls") is None:
            data.pop("pending_tool_calls", None)
        super().__init__(original_input=original_input, **data)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in type(self).model_fields or not hasattr(self, "__pydantic_fields_set__"):
            super().__setattr__(name, value)
            return
        self._apply_updates(**{name: value})

    @model_validator(mode="before")
    @classmethod
    def parse_state_reason(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        raw_state = data.get("state", TaskState.TASK_CREATED)
        state = raw_state if isinstance(raw_state, TaskState) else TaskState(raw_state)
        reason = data.get("state_reason")
        if isinstance(reason, str):
            reason_type = {
                TaskState.TASK_RUNNING: TaskRunningReason,
                TaskState.TASK_PAUSED: TaskPauseReason,
                TaskState.TASK_FAILED: TaskFailReason,
            }.get(state)
            if reason_type is not None:
                data["state_reason"] = reason_type(reason)
        return data

    @model_validator(mode="after")
    def validate_state_invariants(self) -> "Task":
        expected_reason_type = {
            TaskState.TASK_RUNNING: TaskRunningReason,
            TaskState.TASK_PAUSED: TaskPauseReason,
            TaskState.TASK_FAILED: TaskFailReason,
        }.get(self.state)
        if self.state_reason is not None and (
            expected_reason_type is None or not isinstance(self.state_reason, expected_reason_type)
        ):
            raise ValueError(f"state reason {self.state_reason} does not match {self.state.value}")
        if self.is_finished() and self.result is None:
            raise ValueError(f"{self.state.value} task requires a result")
        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the task to a dictionary.
        Returns:
            A dictionary representation of the task.
        """
        for message in self.history:
            _strict_json_value(message.thinking_blocks)
        for pending_call in self.pending_tool_calls:
            _strict_json_value(pending_call.output)
        snapshot = _strict_json_value(self.model_dump(mode="python"))
        assert isinstance(snapshot, dict)
        return snapshot

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Task":
        """Restore one task from serialized snapshot data."""
        snapshot = dict(data)
        if snapshot.get("history") is None:
            snapshot["history"] = []
        if snapshot.get("pending_tool_calls") is None:
            snapshot["pending_tool_calls"] = []
        if "schema_version" not in snapshot:
            snapshot["schema_version"] = 1
            if snapshot.get("state") in {
                TaskState.TASK_COMPLETED,
                TaskState.TASK_COMPLETED.value,
                TaskState.TASK_FAILED,
                TaskState.TASK_FAILED.value,
            } and snapshot.get("state_reason") in {
                TaskRunningReason.TASK_RUNNING_CREATE,
                TaskRunningReason.TASK_RUNNING_CREATE.value,
                TaskRunningReason.TASK_RUNNING_RESUME,
                TaskRunningReason.TASK_RUNNING_RESUME.value,
            }:
                snapshot["state_reason"] = None
        return cls.model_validate(snapshot)

    def _apply_updates(self, **updates: Any) -> None:
        """Validate a complete candidate before committing any field updates."""
        validated_updates = dict(updates)
        for name in {"id", "history", "pending_tool_calls"}.intersection(updates):
            field = type(self).model_fields[name]
            validated_updates[name] = TypeAdapter(field.rebuild_annotation()).validate_python(
                updates[name]
            )
        candidate_data = self.model_dump(mode="python")
        candidate_data.update(validated_updates)
        candidate = type(self).model_validate(candidate_data)
        for name in updates:
            super().__setattr__(name, getattr(candidate, name))

    def _transition_to(self, state: TaskState, state_reason: TaskStateReason | None = None) -> None:
        """
        Transition the task to a new state.
        Args:
            state: The new state.
            state_reason: The reason for the state transition.
        """
        if state not in self.TRANSITIONS[self.state]:
            raise InvalidTransition(self.state, state)

        self._apply_updates(state=state, state_reason=state_reason)

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
        if TaskState.TASK_FAILED not in self.TRANSITIONS[self.state]:
            raise InvalidTransition(self.state, TaskState.TASK_FAILED)
        self._apply_updates(
            state=TaskState.TASK_FAILED,
            state_reason=reason,
            result=self.result if self.result is not None else TaskResult(),
        )

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
        self._apply_updates(step_count=step_count, elapsed_seconds=elapsed_seconds)

    def set_result(self, result: TaskResult) -> None:
        """
        Replace the task result.
        Args:
            result: The result produced for this task.
        """
        self.result = result


def _strict_json_value(value: Any) -> Any:
    """Return an exact JSON value or reject values JSON would coerce or lose."""
    if isinstance(value, Enum):
        return _strict_json_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("task snapshots require finite numbers")
        return value
    if isinstance(value, list):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("task snapshot object keys must be strings")
        return {key: _strict_json_value(item) for key, item in value.items()}
    raise ValueError(f"task snapshot value is not JSON-safe: {type(value).__name__}")


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
