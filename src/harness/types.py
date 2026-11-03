from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class AgentKind(Enum):
    CONVERSATIONAL = "conversational"
    TASK = "task"


class OutputMode(Enum):
    STREAM = "stream"
    BUFFERED = "buffered"


class OutputPhase(Enum):
    EXECUTION = "execution"
    FINAL = "final"
    EVENT = "event"


class RunStatus(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    VALIDATING = "validating"
    BUILDING_CONTEXT = "building_context"
    INFERRING = "inferring"
    DECIDING = "deciding"
    AWAITING_APPROVAL = "awaiting_approval"
    CHECKING_SAFETY = "checking_safety"
    EXECUTING_TOOLS = "executing_tools"
    EMITTING = "emitting"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalRequirement(Enum):
    SKIP = "skip"
    NEEDS_APPROVAL = "needs_approval"
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"
    FORBIDDEN = "forbidden"


TERMINAL_STATUSES = {RunStatus.FINISHED, RunStatus.FAILED, RunStatus.CANCELLED}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str = ""
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class McpDependency:
    server: str
    tool: str | None = None


@dataclass(frozen=True)
class PermissionPolicy:
    rules: tuple[object, ...] = ()


@dataclass(frozen=True)
class ToolSchedule:
    concurrency: str = "serial"
    completion: str = "sync"


@dataclass(frozen=True)
class HarnessInput:
    id: str
    role: str
    content: str
    created_at: float


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmOutput:
    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    output: str
    error: str | None = None
    accepted: bool = False


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ExecCommand:
    command: str
    cwd: str | None = None
    kind: str = "exec_command"


@dataclass(frozen=True)
class WriteStdin:
    data: str
    kind: str = "write_stdin"


@dataclass(frozen=True)
class ApplyPatch:
    patch: str
    kind: str = "apply_patch"


@dataclass(frozen=True)
class McpToolCall:
    server: str
    tool: str
    args: Mapping[str, object] = field(default_factory=dict)
    kind: str = "mcp_tool_call"


@dataclass(frozen=True)
class GenericTool:
    name: str
    args: Mapping[str, object] = field(default_factory=dict)
    kind: str = "generic_tool"


ApprovalAction = ExecCommand | WriteStdin | ApplyPatch | McpToolCall | GenericTool


@dataclass(frozen=True)
class ApprovalScope:
    kind: str
    action_id: str | None = None
    tool_name: str | None = None
    args_hash: str | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    action: ApprovalAction
    requirement: ApprovalRequirement
    scope: ApprovalScope


@dataclass(frozen=True)
class GoalUpdate:
    goal: str
    strategy: str


@dataclass
class RunSnapshot:
    run_id: str
    status: RunStatus
    goal: str
    strategy: str
    step: int
    messages: list[Message]
    error: str | None = None
    output: str | None = None
    last_tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    content: str


@dataclass(frozen=True)
class MemoryQuery:
    goal: str
    text: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class EnvHandle:
    kind: str
    id: str


@dataclass(frozen=True)
class LlmContext:
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    memories: tuple[MemoryRecord, ...] = ()
    goal: str = ""
    strategy: str = ""


@dataclass(frozen=True)
class BuildContextArgs:
    state: RunSnapshot
    system_prompt: str
    memories: tuple[MemoryRecord, ...]
    tools: tuple[ToolDefinition, ...]


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    step: int
    status: RunStatus
    kind: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class FinishDecision:
    text: str
    type: str = "finish"


@dataclass(frozen=True)
class ToolsDecision:
    calls: tuple[ToolCall, ...]
    type: str = "tools"


@dataclass(frozen=True)
class FailDecision:
    reason: str
    type: str = "fail"


Decision = FinishDecision | ToolsDecision | FailDecision


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    output: str | None
    error: str | None
    snapshot: RunSnapshot
