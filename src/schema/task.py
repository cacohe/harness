from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TaskState(Enum):
    """Task state."""

    TASK_CREATED = "created"
    TASK_RUNNING = "running"
    TASK_PAUSED = "paused"
    TASK_FAILED = "failed"
    TASK_COMPLETED = "completed"


class TaskRunningReason(Enum):
    """Task running reason."""

    TASK_RUNNING_CREATE = "create"
    TASK_RUNNING_RESUME = "resume"


class TaskPauseReason(Enum):
    """Task pause reason."""

    TASK_PAUSED_WAITING_APPROVAL = "waiting_approval"
    TASK_PAUSED_TOKEN_EXPIRED = "token_expired"
    TASK_PAUSED_UNKNOWN = "unknown"


class TaskFailReason(Enum):
    """Task fail reason."""

    TASK_FAILED_GUARD_ERROR = "guard_error"
    TASK_FAILED_CONTEXT_BUILD_ERROR = "context_build_error"
    TASK_FAILED_LLM_ERROR = "llm_error"
    TASK_FAILED_TOOL_ERROR = "tool_error"
    TASK_FAILED_SKILL_ERROR = "skill_error"
    TASK_FAILED_TIMEOUT = "timeout"
    TASK_FAILED_MAX_STEPS = "max_steps"
    TASK_FAILED_CANCEL = "cancel"
    TASK_FAILED_UNKNOWN = "unknown"


TaskStateReason = TaskRunningReason | TaskPauseReason | TaskFailReason


class TaskPlan(BaseModel):
    """Task plan."""

    goal: list[str]
    completed: list[str] = Field(default_factory=list)
    current: str | None = None
    remaining: list[str] = Field(default_factory=list)
    clarification: str | None = Field(
        default=None,
        description=(
            "A question for the user when you decide not to plan yet. "
            "Leave empty when you provide a plan."
        ),
    )

    def to_dict(self) -> dict:
        """
        Convert the task plan to a dictionary.
        Returns:
            The plan fields.
        """
        return self.model_dump()

    def advance_plan(self, current: str):
        """
        Advance the task plan.
        """
        ...


class TaskResult(BaseModel):
    """Task result."""

    output: str = ""

    def __init__(self, output: str | None = ""):
        super().__init__(output=output)

    @field_validator("output", mode="before")
    @classmethod
    def normalize_missing_output(cls, value):
        return "" if value is None else value

    def to_dict(self) -> dict:
        """
        Convert the task result to a dictionary.
        Returns:
            The result text.
        """
        return self.model_dump(mode="json")
