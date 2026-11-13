from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field


@dataclass
class Tool:
    """
    A tool in the harness.
    Args:
        id: The unique identifier of the tool.
        name: The name of the tool.
        description: The description of the tool.
        input_schema: The input schema of the tool.
        executer: The executer of the tool.
    """

    id: str
    name: str
    description: str
    input_schema: dict
    executer: Callable


class ToolExecution(BaseModel):
    """
    A tool execution.
    Args:
        tool: The tool to execute.
        inputs: The inputs to the tool.
    """

    tool: Tool
    inputs: dict


class ApprovalDecision(Enum):
    APPROVAL_DECISION_APPROVED = "approved"
    APPROVAL_DECISION_REJECTED = "rejected"
    APPROVAL_DECISION_TIMEOUT = "timeout"


class ToolExecutionState(Enum):
    TOOL_EXECUTION_STATE_PENDING = "pending"
    TOOL_EXECUTION_STATE_COMPLETED = "completed"
    TOOL_EXECUTION_STATE_FAILED = "failed"


class ToolExecutionFailReason(Enum):
    TOOL_EXECUTION_FAIL_REASON_TIMEOUT = "timeout"
    TOOL_EXECUTION_FAIL_REASON_MAX_STEPS = "max_steps"
    TOOL_EXECUTION_FAIL_REASON_CANCELLED = "cancelled"
    TOOL_EXECUTION_FAIL_REASON_REJECTED = "rejected"
    TOOL_EXECUTION_FAIL_REASON_UNKNOWN = "unknown"
    TOOL_EXECUTION_FAIL_REASON_NOT_SUPPORTED = "not_supported"
    TOOL_EXECUTION_FAIL_REASON_NOT_AUTHENTICATED = "not_authenticated"


class ToolExecutionResult(BaseModel):
    tool_id: str
    tool_name: str
    state: ToolExecutionState = ToolExecutionState.TOOL_EXECUTION_STATE_PENDING
    output: Any | None = None
    fail_reason: ToolExecutionFailReason | None = None
    arguments: str | None = None
    approval: ApprovalDecision | None = None
    needs_approval: bool = False

    def can_run(self) -> bool:
        return (
            not self.needs_approval or self.approval == ApprovalDecision.APPROVAL_DECISION_APPROVED
        )

    def is_completed(self) -> bool:
        return self.state == ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED

    def is_failed(self) -> bool:
        return self.state == ToolExecutionState.TOOL_EXECUTION_STATE_FAILED

    def is_pending(self) -> bool:
        return self.state == ToolExecutionState.TOOL_EXECUTION_STATE_PENDING


class ToolCallResult(BaseModel):
    pending_tools: list[ToolExecutionResult] = Field(default_factory=list)
    completed_tools: list[ToolExecutionResult] = Field(default_factory=list)
    failed_tools: list[ToolExecutionResult] = Field(default_factory=list)
