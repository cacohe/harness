from pydantic import BaseModel, Field

from src.core.llm import LlmCallError
from src.core.task import TaskPlan


class TaskPlanSchema(BaseModel):
    """Structured plan returned by the planning call."""

    goal: list[str]
    current: str | None = None
    remaining: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)


def task_plan_from_structured(structured) -> TaskPlan:
    """
    Convert a planning response into a task plan.
    Args:
        structured: A TaskPlanSchema, or a dict with the same fields.
    Returns:
        The task plan stored on the task.
    Raises:
        LlmCallError: The planning response is not a plan.
    """
    data = _plan_data(structured)
    goal = data.get("goal") or []
    if isinstance(goal, str):
        goal = [goal]
    return TaskPlan(
        goal=list(goal),
        completed=list(data.get("completed") or []),
        current=data.get("current"),
        remaining=list(data.get("remaining") or []),
    )


def _plan_data(structured) -> dict:
    if isinstance(structured, BaseModel):
        return structured.model_dump()
    if isinstance(structured, dict):
        return structured
    raise LlmCallError("planning did not return a task plan")
