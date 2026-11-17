from typing import Protocol, runtime_checkable

from src.core.task import Task
from src.schema.context import Context, Message
from src.schema.llm import ROLE_ASSISTANT, ROLE_TOOL, LLMOutput
from src.schema.tool import ToolCallResult, ToolExecutionResult

PENDING_TOOL_MESSAGE = "The tool was not executed because it is waiting for approval."
HELD_TOOL_MESSAGE = (
    "The tool was not executed because an earlier tool call is waiting for approval."
)


class ContextBuilderError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


@runtime_checkable
class ContextBuilder(Protocol):
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
    items = result.ordered_tools or [
        *result.completed_tools,
        *result.failed_tools,
        *result.pending_tools,
    ]
    return [build_tool_message(item) for item in items]


def build_tool_message(item: ToolExecutionResult) -> Message:
    """
    Build the tool message for one tool execution.
    Args:
        item: A completed, failed, or pending tool execution.
    Returns:
        The tool output, the failure text, or a note that the call waits for approval.
    """
    if item.is_pending():
        content = PENDING_TOOL_MESSAGE if item.needs_approval else HELD_TOOL_MESSAGE
    elif item.output is not None:
        content = str(item.output)
    elif item.is_failed() and item.fail_reason is not None:
        content = item.fail_reason.value
    else:
        content = ""
    return Message(
        role=ROLE_TOOL,
        content=content,
        tool_call_id=item.tool_id,
        name=item.tool_name,
    )
