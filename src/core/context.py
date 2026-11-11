from abc import ABC, abstractmethod

from src.core.task import Task
from src.schema.context import Context, Message
from src.schema.llm import ROLE_ASSISTANT, ROLE_TOOL, LLMOutput
from src.schema.tool import ToolCallResult

DEFAULT_SYSTEM_PROMPT_FOR_PLANNING = """
You are an expert in task management, skilled at converting vague inputs into clear task objectives, and designing reasonable execution steps based on those task objectives.
"""
DEFAULT_SYSTEM_PROMPT_FOR_INFER = """
You are an expert in task execution, skilled at executing tasks based on the given task objectives and execution steps.
"""
DEFAULT_SYSTEM_PROMPT_FOR_FINAL = """
Answer the original goal from the results above. Do not call tools. Based on the current task state, explain the execution status: what was done, what remains, and why it was not completed.
"""


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
    def build_infer_context(
        self, system_prompt: str, task: Task, tools: list, skills: list, **kwargs
    ) -> Context:
        """
        Build the context for the step.
        Args:
            system_prompt: The system prompt.
            task: The task for the step. Its history is the prior transcript.
            tools: Tools available for this call.
            skills: Skills available for this call.
            **kwargs: Additional keyword arguments.
        Returns:
            Messages for one execution call, and the tools available for that call.
        """
        ...
    
    @abstractmethod
    def build_final_context(self, system_prompt: str, task: Task, **kwargs) -> Context:
        """
        Build the context for the final.
        Args:
            system_prompt: The system prompt of the harness.
            task: The task for the step.
            **kwargs: Additional keyword arguments.
        Returns:
            The execution messages, a closing instruction, and no callable tools.
        """
        ...


def build_history_message(llm_output: LLMOutput) -> Message:
    return Message(
        role=ROLE_ASSISTANT,
        content=llm_output.content,
        tool_calls=list(llm_output.tool_calls),
        reasoning_content=llm_output.reasoning_content,
        thinking_blocks=llm_output.thinking_blocks,
    )


def build_tool_messages(result: ToolCallResult) -> list[Message]:
    """
    Build the tool messages for one batch of tool results.
    Args:
        result: Completed and failed tool executions from one batch.
    Returns:
        One tool message per completed or failed execution.
    """
    messages = []
    for item in result.completed_tools:
        messages.append(
            Message(
                role=ROLE_TOOL,
                content="" if item.output is None else str(item.output),
                tool_call_id=item.tool_id,
                name=item.tool_name,
            )
        )
    for item in result.failed_tools:
        if item.output is not None:
            content = str(item.output)
        else:
            content = item.fail_reason.value if item.fail_reason else ""
        messages.append(
            Message(
                role=ROLE_TOOL,
                content=content,
                tool_call_id=item.tool_id,
                name=item.tool_name,
            )
        )
    for item in result.pending_tools:
        messages.append(
            Message(
                role=ROLE_TOOL,
                content="The tool was not executed because it is waiting for approval.",
                tool_call_id=item.tool_id,
                name=item.tool_name,
            )
        )
    return messages
