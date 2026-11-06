import json
from dataclasses import dataclass, field

from src.core.context import Message
from src.core.llm import ROLE_TOOL, LlmToolCall
from src.core.tool import ToolExecutor, ToolResult


@dataclass
class ToolCallBatch:
    """Tool calls finished so far, and any calls still waiting for approval."""

    messages: list[Message] = field(default_factory=list)
    pending: list[LlmToolCall] = field(default_factory=list)

    @property
    def waiting_for_approval(self) -> bool:
        return bool(self.pending)


def collect_tool_results(executor: ToolExecutor, calls: list[LlmToolCall]) -> ToolCallBatch:
    """
    Run each tool call until one needs approval.
    Args:
        executor: The executor that runs a single tool.
        calls: Tool calls from one model output, in order.
    Returns:
        Completed tool messages, and the calls that were not run because approval is required.
    """
    messages = []
    for index, call in enumerate(calls):
        if needs_approval(executor, call.name):
            return ToolCallBatch(messages, calls[index:])
        messages.append(tool_message(executor, call))
    return ToolCallBatch(messages)


def needs_approval(executor: ToolExecutor, tool_name: str) -> bool:
    """
    Check whether the executor requires approval before this tool runs.
    Args:
        executor: The executor that owns the approval rule.
        tool_name: The tool name requested by the model.
    Returns:
        True only when the executor explicitly requires approval.
    """
    check = getattr(executor, "is_need_approval", None)
    return callable(check) and check(tool_name) is True


def tool_message(executor: ToolExecutor, call: LlmToolCall) -> Message:
    """
    Run one tool call and return the message the model should see.
    Args:
        executor: The executor that runs the tool.
        call: The tool call requested by the model.
    Returns:
        A tool message. Execution and argument errors are the message content.
    """
    try:
        result = executor.execute(call.name, parse_tool_arguments(call.arguments))
        content = tool_result_text(result)
    except Exception as error:
        content = f"Error: {error}"
    return Message(role=ROLE_TOOL, content=content, tool_call_id=call.id, name=call.name)


def parse_tool_arguments(arguments) -> dict:
    """
    Parse model tool arguments into the dict an executor accepts.
    Args:
        arguments: A JSON object string, or a dict.
    Returns:
        The argument dict. An empty value becomes an empty dict.
    Raises:
        ValueError: The text is not a JSON object.
    """
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    value = json.loads(arguments)
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be a JSON object")
    return value


def tool_result_text(result) -> str:
    """
    Read the text that should be sent back to the model.
    Args:
        result: A ToolResult, or any other value returned by an executor.
    Returns:
        The success content, the error text, or the string form of another result.
    """
    if isinstance(result, ToolResult):
        if result.is_success:
            return result.content or ""
        return result.error or ""
    return "" if result is None else str(result)
