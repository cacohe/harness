import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from src.core.errors import RunFailure
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId


@dataclass(frozen=True)
class RunLimits:
    max_steps: int
    max_elapsed_seconds: float
    max_inline_result_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_inline_result_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        elapsed = self.max_elapsed_seconds
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise ValueError("max_elapsed_seconds must be a positive finite number")
        try:
            finite = math.isfinite(elapsed)
        except OverflowError:
            finite = False
        if not finite or elapsed <= 0:
            raise ValueError("max_elapsed_seconds must be a positive finite number")


@dataclass(frozen=True)
class CommandContext:
    timestamp: datetime
    elapsed_seconds: float
    event_ids: tuple[EventId, ...] = ()

    def __post_init__(self) -> None:
        elapsed = self.elapsed_seconds
        if isinstance(elapsed, bool) or type(elapsed) not in (int, float) or elapsed < 0:
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        try:
            finite = math.isfinite(elapsed)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("elapsed_seconds must be a non-negative finite number")
        if type(self.event_ids) is not tuple or not all(
            isinstance(event_id, EventId) for event_id in self.event_ids
        ):
            raise ValueError("event_ids must be a tuple of EventId values")


@dataclass(frozen=True)
class _RunCommand:
    run_id: RunId
    session_id: SessionId
    context: CommandContext = field(kw_only=True)


@dataclass(frozen=True)
class SubmitRun(_RunCommand):
    submission_id: SubmissionId
    profile_id: ProfileId
    user_input: str


@dataclass(frozen=True)
class ResumeRun(_RunCommand):
    pass


@dataclass(frozen=True)
class CancelRun(_RunCommand):
    pass


@dataclass(frozen=True)
class DecideToolCall(_RunCommand):
    tool_call_id: str
    approved: bool


@dataclass(frozen=True)
class StartStep(_RunCommand):
    step_id: str


@dataclass(frozen=True)
class FinishStep(_RunCommand):
    step_id: str


@dataclass(frozen=True)
class StartModelCall(_RunCommand):
    step_id: str
    model_call_id: str


@dataclass(frozen=True)
class FinishModelCall(_RunCommand):
    step_id: str
    model_call_id: str
    result: str
    tool_call_id: str | None = None


@dataclass(frozen=True)
class StartToolCall(_RunCommand):
    step_id: str
    model_call_id: str
    tool_call_id: str
    requires_approval: bool


@dataclass(frozen=True)
class FinishToolCall(_RunCommand):
    step_id: str
    model_call_id: str
    tool_call_id: str
    result: str | RunFailure


@dataclass(frozen=True)
class CompleteRun(_RunCommand):
    final_text: str


@dataclass(frozen=True)
class FailRun(_RunCommand):
    failure: RunFailure


CoreCommand: TypeAlias = (
    SubmitRun
    | ResumeRun
    | CancelRun
    | DecideToolCall
    | StartStep
    | FinishStep
    | StartModelCall
    | FinishModelCall
    | StartToolCall
    | FinishToolCall
    | CompleteRun
    | FailRun
)
