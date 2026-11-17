import math
from dataclasses import dataclass

from src.core.ids import ProfileId, RunId, SessionId, SubmissionId


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
class _RunCommand:
    run_id: RunId
    session_id: SessionId


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
