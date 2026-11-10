import json
from abc import ABC, abstractmethod

from src.core.sandbox import Sandbox
from src.logger import get_logger
from src.schema.llm import LlmToolCall
from src.schema.tool import (
    ApprovalDecision,
    Tool,
    ToolCallResult,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)

logger = get_logger("src.core.tool")


class ToolRegistry(ABC):
    @abstractmethod
    def register(self, tool: Tool): ...

    @abstractmethod
    def unregister(self, tool_id: str): ...

    @abstractmethod
    def get_tool(self, tool_id: str) -> Tool: ...

    @abstractmethod
    def get_all_tools(self) -> list[Tool]: ...


class ToolExecutor(ABC):
    @abstractmethod
    def is_need_approval(self, tool: Tool) -> bool:
        """
        Check if the tool needs approval.
        Args:
            tool: The tool to check.
        Returns:
            True if the tool needs approval, False otherwise.
        """
        ...

    @abstractmethod
    def launch_approval(self, tool: Tool) -> bool:
        """
        Launch the approval process.
        Args:
            tool: The tool to launch the approval process for.
        Returns:
            True if the approval process is launched, False otherwise.
        """
        ...

    @abstractmethod
    def is_need_sandbox(self, tool: Tool) -> bool:
        """
        Check if the tool needs sandbox.
        Args:
            tool: The tool to check.
        Returns:
            True if the tool needs sandbox, False otherwise.
        """
        ...

    @abstractmethod
    def get_sandbox(self) -> Sandbox:
        """
        Get the sandbox for a tool.
        Args:

        Returns:
            The sandbox for the tool.
        """
        ...

    @abstractmethod
    def find_tool(self, name: str) -> Tool | None:
        """
        Find a registered tool for a model tool call.
        Args:
            name: The tool name from the model.
        Returns:
            The matching tool, or None when no tool has that name.
        """
        ...

    def execute_with_sandbox(
        self, tool_id: str, inputs: dict, sandbox: Sandbox
    ) -> ToolExecutionResult:
        """
        Execute a tool with a sandbox.
        Args:
            tool_id: The id of the tool.
            inputs: The inputs to the tool.
            sandbox: The sandbox to use for the tool execution.
        Returns:
            The result of the tool execution.
        """
        ...

    @abstractmethod
    def execute_without_sandbox(self, tool_id: str, inputs: dict) -> ToolExecutionResult:
        """
        Execute a tool without a sandbox.
        Args:
            tool_id: The id of the tool.
            inputs: The inputs to the tool.
        Returns:
            The result of the tool execution.
        """
        ...

    @abstractmethod
    def execute(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        """
        Execute a tool.
        Args:
            tool: The tool to execute.
            inputs: The inputs to the tool.
        Returns:
            The result of the tool execution.
        """
        ...
        tool_id = tool.id
        tool_name = tool.name

        if self.is_need_approval(tool) is True:
            approval_decision = self.launch_approval(tool)

            if approval_decision == ApprovalDecision.APPROVAL_DECISION_REJECTED:
                return ToolExecutionResult(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                    output=None,
                    fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED,
                )
            elif approval_decision == ApprovalDecision.APPROVAL_DECISION_TIMEOUT:
                return ToolExecutionResult(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                    output=None,
                    fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_TIMEOUT,
                )

        if self.is_need_sandbox(tool) is True:
            sandbox = self.get_sandbox()
            return self.execute_with_sandbox(tool, inputs, sandbox)

        return self.execute_without_sandbox(tool, inputs)

    def execute_tool_calls(self, tool_calls: list[LlmToolCall]) -> ToolCallResult:
        """
        Run model tool calls.
        Args:
            tool_calls: Tool calls from the model, in order. Each call has id, name, and arguments.
        Returns:
            Completed, failed, and pending tool executions.
        """
        pending_tools = []
        completed_tools = []
        failed_tools = []

        for call in tool_calls:
            tool_execution_result = self._execute_model_call(call)
            if tool_execution_result.is_completed():
                logger.info(
                    "tool finished",
                    extra={"context": {"tool": tool_execution_result.tool_name}},
                )
                completed_tools.append(tool_execution_result)
            elif tool_execution_result.is_failed():
                logger.warning(
                    "tool failed",
                    extra={
                        "context": {
                            "tool": tool_execution_result.tool_name,
                            "reason": (
                                tool_execution_result.fail_reason.value
                                if tool_execution_result.fail_reason
                                else None
                            ),
                        }
                    },
                )
                failed_tools.append(tool_execution_result)
            else:
                pending_tools.append(tool_execution_result)

        return ToolCallResult(
            pending_tools=pending_tools,
            completed_tools=completed_tools,
            failed_tools=failed_tools,
        )

    def _execute_model_call(self, call: LlmToolCall) -> ToolExecutionResult:
        try:
            parsed = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                output="tool arguments are not a JSON object",
                fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_UNKNOWN,
            )
        tool = self.find_tool(call.name)
        if tool is None:
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                output=None,
                fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_NOT_SUPPORTED,
            )
        result = self.execute(tool, parsed)
        result.tool_id = call.id
        return result
