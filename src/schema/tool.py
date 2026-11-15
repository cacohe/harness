from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field, InstanceOf

if TYPE_CHECKING:
    from src.core.workspace import Workspace

TOOL_CALL_CANCELLED_OUTPUT = "Tool call cancelled."


class ToolOrigin(str, Enum):
    LOCAL = "local"
    MCP = "mcp"
    CUSTOM = "custom"


class SandboxKind(str, Enum):
    """Isolation capability required to execute a sandboxed tool."""

    COMMAND = "command"


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
    requires_sandbox: bool = False
    input_validator: Callable[[dict], dict] | None = None
    workspace: Workspace | None = None
    origin: ToolOrigin = ToolOrigin.CUSTOM
    requires_approval: bool = False
    approval_scope: str | None = None
    sandbox_kind: SandboxKind | None = None


class ToolExecution(BaseModel):
    """
    A tool execution.
    Args:
        tool: The tool to execute.
        inputs: The inputs to the tool.
    """

    tool: InstanceOf[Tool]
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
    TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION = "workspace_violation"
    TOOL_EXECUTION_FAIL_REASON_SANDBOX_REQUIRED = "sandbox_required"
    TOOL_EXECUTION_FAIL_REASON_REMOTE_ERROR = "remote_error"
    TOOL_EXECUTION_FAIL_REASON_APPROVAL_STALE = "approval_stale"


class ToolExecutionResult(BaseModel):
    tool_id: str
    tool_name: str
    state: ToolExecutionState = ToolExecutionState.TOOL_EXECUTION_STATE_PENDING
    output: Any | None = None
    fail_reason: ToolExecutionFailReason | None = None
    arguments: str | None = None
    approval: ApprovalDecision | None = None
    needs_approval: bool = False
    workspace_id: str | None = None
    approval_scope: str | None = None

    def can_run(self) -> bool:
        if self.approval in {
            ApprovalDecision.APPROVAL_DECISION_REJECTED,
            ApprovalDecision.APPROVAL_DECISION_TIMEOUT,
        }:
            return False
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
    ordered_tools: list[ToolExecutionResult] = Field(default_factory=list)
