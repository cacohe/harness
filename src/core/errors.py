from dataclasses import dataclass, field
from enum import StrEnum


class RunErrorCode(StrEnum):
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    MODEL_FAILED = "model_failed"
    TOOL_FAILED = "tool_failed"
    TOOL_TIMED_OUT = "tool_timed_out"
    ARTIFACT_FAILED = "artifact_failed"
    JOURNAL_CONFLICT = "journal_conflict"
    CANCELLED = "cancelled"
    LIMIT_EXCEEDED = "limit_exceeded"
    INVARIANT_VIOLATION = "invariant_violation"


_PUBLIC_MESSAGES = {
    RunErrorCode.CONTEXT_BUDGET_EXCEEDED: "The required context exceeds the input budget.",
    RunErrorCode.MODEL_FAILED: "Model request failed.",
    RunErrorCode.TOOL_FAILED: "Tool call failed.",
    RunErrorCode.TOOL_TIMED_OUT: "Tool call timed out.",
    RunErrorCode.ARTIFACT_FAILED: "Artifact operation failed.",
    RunErrorCode.JOURNAL_CONFLICT: "The run changed before it could be saved.",
    RunErrorCode.CANCELLED: "The run was cancelled.",
    RunErrorCode.LIMIT_EXCEEDED: "The run exceeded its execution limits.",
    RunErrorCode.INVARIANT_VIOLATION: "The requested run state is invalid.",
}


@dataclass(frozen=True)
class RunFailure:
    code: RunErrorCode
    message: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.code, RunErrorCode):
            raise ValueError("Run failure requires a stable error code")
        object.__setattr__(self, "message", _PUBLIC_MESSAGES[self.code])
