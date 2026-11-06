import uuid
from enum import Enum
from typing import FrozenSet, Optional

from src.core.skills import Skill
from src.core.tool import Tool


class TaskState(Enum):
    """Task state."""

    TASK_CREATED = "created"
    TASK_QUEUE = "queue"
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


class TaskPlan:
    """Task plan."""

    goal: list[str]
    completed: Optional[list[str]] = None
    current: Optional[str] = None
    remaining: Optional[list[str]] = None

    def __init__(
        self,
        goal: list[str],
        completed: Optional[list[str]] = None,
        current: Optional[str] = None,
        remaining: Optional[list[str]] = None,
    ):
        self.goal = goal
        self.completed = completed
        self.current = current
        self.remaining = remaining

    def to_dict(self) -> dict:
        """
        Convert the task progress to a dictionary.
        Returns:
            A dictionary representation of the task progress.
        """
        return {
            "goal": self.goal,
            "completed": self.completed,
            "current": self.current,
            "remaining": self.remaining,
        }

    def advance_plan(self, current: str):
        """
        Advance the task plan.
        """
        ...


class InvalidTransition(Exception):
    """Invalid transition exception."""

    def __init__(self, current_state: TaskState, new_state: TaskState) -> None:
        super().__init__(f"Invalid transition: {current_state} -> {new_state}")


class Task:
    """A task to be executed by the harness."""

    id: str
    original_input: str
    state: TaskState
    state_reason: Optional[TaskStateReason] = None
    plan: TaskPlan

    TRANSITIONS: dict[TaskState, FrozenSet[TaskState]] = {
        TaskState.TASK_CREATED: frozenset(
            [TaskState.TASK_QUEUE, TaskState.TASK_RUNNING, TaskState.TASK_FAILED]
        ),
        TaskState.TASK_QUEUE: frozenset([TaskState.TASK_RUNNING]),
        TaskState.TASK_RUNNING: frozenset(
            [TaskState.TASK_PAUSED, TaskState.TASK_FAILED, TaskState.TASK_COMPLETED]
        ),
        TaskState.TASK_PAUSED: frozenset([TaskState.TASK_QUEUE, TaskState.TASK_RUNNING]),
        TaskState.TASK_FAILED: frozenset([]),
        TaskState.TASK_COMPLETED: frozenset([]),
    }

    def __init__(self, original_input: str, tools: list[Tool], skills: list[Skill]):
        self.id = uuid.uuid4()
        self.original_input = original_input
        self.tools = tools
        self.skills = skills
        self.state = TaskState.TASK_CREATED
        self.state_reason = None
        self.plan = TaskPlan(goal=[], completed=[], current=None, remaining=[])
        self.history = []
        self.pending_tool_calls = []
        self.result = None

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
            "result": self.result,
        }

    def set_plan(self, plan: TaskPlan) -> None:
        """
        Replace the task plan.
        Args:
            plan: The plan produced for this task.
        """
        self.plan = plan

    def transition_to(
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

    def is_complete(self) -> bool:
        """
        Check if the task is complete.
        Returns:
            True if the task is complete, False otherwise.
        """
        return self.state == TaskState.TASK_COMPLETED

    def is_failed(self) -> bool:
        """
        Check if the task is failed.
        Returns:
            True if the task is failed, False otherwise.
        """
        return self.state == TaskState.TASK_FAILED

    def persist(self) -> None:
        """
        Persist the task.
        """
        ...
