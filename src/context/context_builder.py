from src.core.context import (
    DEFAULT_SYSTEM_PROMPT_FOR_INFER,
    DEFAULT_SYSTEM_PROMPT_FOR_PLANNING,
    ContextBuilder,
    ContextBuilderError,
)
from src.core.task import Task
from src.schema.context import Context, Message
from src.schema.skills import Skill
from src.schema.tool import Tool


class DefaultContextBuilder(ContextBuilder):
    """Build the planning prompt and the execution transcript."""

    def __init__(self): ...

    def build_planning_context(self, system_prompt: str, task: Task, **kwargs) -> Context:
        """
        Build the context for planning.
        Args:
            system_prompt: Extra instructions from the harness.
            task: The task to plan.
            **kwargs: Unused. Accepted so callers can pass the same arguments as inference.
        Returns:
            A system instruction, the original request, and no tools.
        """
        try:
            return Context(
                messages=[
                    Message(
                        role="system",
                        content=_with_extra_prompt(
                            DEFAULT_SYSTEM_PROMPT_FOR_PLANNING, system_prompt
                        ),
                    ),
                    Message(role="user", content=task.original_input),
                ],
                tools=[],
            )
        except Exception as error:
            raise ContextBuilderError(f"Failed to build planning context: {error}") from error

    def build_infer_context(
        self,
        system_prompt: str,
        task: Task,
        tools: list[Tool],
        skills: list[Skill],
        **kwargs,
    ) -> Context:
        """
        Build the context for one execution step.
        Args:
            system_prompt: Extra instructions from the harness.
            task: The task being executed. Its history is the prior transcript.
            tools: Tools supplied by the harness for this call.
            skills: Skills supplied by the harness for this call.
            **kwargs: Unused.
        Returns:
            The system instruction, the original request, history, the current plan,
            and the tools for this call.
        """
        try:
            messages = [
                Message(role="system", content=_infer_system_prompt(system_prompt, skills)),
                Message(role="user", content=task.original_input),
                *list(task.history or []),
            ]
            plan = _plan_message(task)
            if plan is not None:
                messages.append(plan)
            return Context(messages=messages, tools=list(tools))
        except Exception as error:
            raise ContextBuilderError(f"Failed to build infer context: {error}") from error


def _with_extra_prompt(base: str, extra: str) -> str:
    text = base.strip()
    addition = (extra or "").strip()
    if not addition:
        return text
    return f"{text}\n\n{addition}"


def _infer_system_prompt(system_prompt: str, skills: list[Skill]) -> str:
    text = _with_extra_prompt(DEFAULT_SYSTEM_PROMPT_FOR_INFER, system_prompt)
    formatted = _format_skills(skills)
    if not formatted:
        return text
    return f"{text}\n\n{formatted}"


def _format_skills(skills) -> str:
    if not skills:
        return ""
    lines = ["Skills:"]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description}\n{skill.content}")
    return "\n".join(lines)


def _plan_message(task: Task) -> Message | None:
    plan = task.plan
    if not _plan_has_work(plan):
        return None
    return Message(
        role="user",
        content="\n".join(
            [
                f"Goal: {plan.goal}",
                f"Current: {plan.current}",
                f"Completed: {plan.completed}",
                f"Remaining: {plan.remaining}",
            ]
        ),
    )


def _plan_has_work(plan) -> bool:
    return bool(plan.goal or plan.current or plan.remaining or plan.completed)
