from typing import Any

from src.core.context import ContextBuilderError
from src.core.task import Task
from src.schema.context import Context, Message
from src.schema.skills import Skill
from src.schema.task import TaskPlan
from src.schema.tool import Tool

DEFAULT_SYSTEM_PROMPT_FOR_PLANNING = """
You are an expert in task management, skilled at converting inputs into clear task objectives and designing reasonable execution steps based on those objectives.
You decide whether this input needs a plan. Write the goal and steps when you plan. Use clarification when you need to ask the user something instead.
"""
DEFAULT_SYSTEM_PROMPT_FOR_INFER = """
You are an expert in task execution, skilled at executing tasks based on the given task objectives and execution steps.
"""
DEFAULT_SYSTEM_PROMPT_FOR_FINAL = """
Answer the original goal from the results above. Do not call tools. Based on the current task state, explain the execution status: what was done, what remains, and why it was not completed.
"""


class DefaultContextBuilder:
    """Build the planning prompt and the execution transcript."""

    def build_planning_context(
        self, system_prompt: str, task: Task, tools: list[Tool] | None = None, **kwargs
    ) -> Context:
        """
        Build the context for planning.
        Args:
            system_prompt: Extra instructions from the harness.
            task: The task to plan.
            tools: Tools the execution loop can call. Their names are listed for the planner.
            **kwargs: Unused. Accepted so callers can pass the same arguments as inference.
        Returns:
            A system instruction that names the available tools, the original request,
            and no callable tools.
        """
        try:
            return Context(
                messages=[
                    Message(
                        role="system",
                        content=_planning_system_prompt(system_prompt, tools),
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
            messages: list[dict[str, Any] | Message] = [
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

    def build_final_context(self, system_prompt: str, task: Task, **kwargs) -> Context:
        """
        Build the context for one closing model call.
        Args:
            system_prompt: Extra instructions from the harness.
            task: The task being closed. Its history is the prior transcript and is left unchanged.
            **kwargs: tools and skills used to assemble the execution transcript. Tools are not offered.
        Returns:
            The execution messages, a closing instruction, and no callable tools.
        """
        tools = list(kwargs.get("tools") or [])
        skills = list(kwargs.get("skills") or [])
        context = self.build_infer_context(system_prompt, task, tools, skills)
        context.tools = []
        context.messages.append(
            Message(role="user", content=DEFAULT_SYSTEM_PROMPT_FOR_FINAL.strip())
        )
        return context


def _with_extra_prompt(base: str, extra: str) -> str:
    text = base.strip()
    addition = (extra or "").strip()
    if not addition:
        return text
    return f"{text}\n\n{addition}"


def _planning_system_prompt(system_prompt: str, tools: list[Tool] | None) -> str:
    text = _with_extra_prompt(DEFAULT_SYSTEM_PROMPT_FOR_PLANNING, system_prompt)
    catalog = _format_tools(tools)
    if not catalog:
        return text
    return f"{text}\n\n{catalog}"


def _format_tools(tools: list[Tool] | None) -> str:
    if not tools:
        return ""
    lines = ["Available tools:"]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)


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
    if plan is None or not _plan_has_work(plan):
        return None
    lines = [f"Goal: {plan.goal}"]
    if not task.history:
        lines.extend(
            [
                f"Current: {plan.current}",
                f"Completed: {plan.completed}",
                f"Remaining: {plan.remaining}",
            ]
        )
    return Message(role="user", content="\n".join(lines))


def _plan_has_work(plan: TaskPlan) -> bool:
    return bool(plan.goal or plan.current or plan.remaining or plan.completed)
