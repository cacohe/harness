from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.task import Task

DEFAULT_SYSTEM_PROMPT_FOR_PLANNING = """
You are an expert in task management, skilled at converting vague inputs into clear task objectives, and designing reasonable execution steps based on those task objectives.
"""
DEFAULT_SYSTEM_PROMPT_FOR_INFER = """
You are an expert in task execution, skilled at executing tasks based on the given task objectives and execution steps.
"""


@dataclass
class Message:
    """One chat message replayed on the next model call."""

    role: str
    content: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list | None = None


@dataclass
class Context:
    """One LiteLLM call: ordered messages with unified roles, and the tool catalog."""

    messages: list[Message]
    tools: list


class ContextBuilderError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ContextBuilder(ABC):
    def __init__(self): ...

    @abstractmethod
    def build_planning_context(self, system_prompt: str, task: Task, **kwargs) -> Context:
        """
        Build the context for the planning.
        Args:
            system_prompt: The system prompt of the harness.
            task: The task for the step.
            **kwargs: Additional keyword arguments.
        Returns:
            Messages for one planning call, and an empty tool catalog.
        """
        ...

    @abstractmethod
    def build_infer_context(self, system_prompt: str, task: Task, **kwargs) -> Context:
        """
        Build the context for the step.
        Args:
            system_prompt: The system prompt.
            task: The task for the step.
            **kwargs: Additional keyword arguments. history is the prior transcript.
        Returns:
            Messages for one execution call, and the tools available for that call.
        """
        ...
