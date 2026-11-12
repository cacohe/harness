from src.core.sandbox import Sandbox
from src.core.tool import ToolExecutor, ToolRegistry
from src.schema.tool import (
    Tool,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)

_NEED_APPROVAL_TOOLS = {"shell", "write_file", "edit_file"}


class DefaultToolExecutor(ToolExecutor):
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def is_need_approval(self, tool: Tool) -> bool:
        """
        Check if the tool needs approval.
        Args:
            tool: The tool to check.
        Returns:
            True when the tool can run commands or change files.
        """
        return tool.name in _NEED_APPROVAL_TOOLS

    def launch_approval(self, tool: Tool) -> bool:
        """
        Launch the approval process.
        Args:
            tool: The tool to launch the approval process for.
        Returns:
            True if the approval process is launched, False otherwise.
        """
        return False

    def is_need_sandbox(self, tool: Tool) -> bool:
        """
        Check if the tool needs sandbox.
        Args:
            tool: The tool to check.
        Returns:
            True if the tool needs sandbox, False otherwise.
        """
        return False

    def find_tool(self, name: str) -> Tool | None:
        """
        Find a registered tool by name or id.
        Args:
            name: The tool name from the model.
        Returns:
            The matching tool, or None when no tool has that name.
        """
        for tool in self.tool_registry.get_all_tools():
            if tool.name == name or tool.id == name:
                return tool
        return None

    def get_sandbox(self) -> Sandbox:
        """
        Get the sandbox for a tool.
        Returns:
            The sandbox for the tool.
        """
        raise NotImplementedError

    def execute_without_sandbox(self, tool_id: str, inputs: dict) -> ToolExecutionResult:
        """
        Execute a registered tool outside a sandbox.
        Args:
            tool_id: The id of the tool.
            inputs: The inputs to the tool.
        Returns:
            The completed tool execution.
        """
        tool = self.tool_registry.get_tool(tool_id)
        try:
            output = tool.executer(inputs)
        except Exception as error:
            return ToolExecutionResult(
                tool_id=tool.id,
                tool_name=tool.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                output=f"{type(error).__name__}: {error}",
                fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_UNKNOWN,
            )
        return ToolExecutionResult(
            tool_id=tool.id,
            tool_name=tool.name,
            state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
            output=output,
        )

    def execute(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        """
        Execute a tool.
        Args:
            tool: The tool to execute.
            inputs: The inputs to the tool.
        Returns:
            The result of the tool execution.
        """
        if self.is_need_sandbox(tool) is True:
            return self.execute_with_sandbox(tool.id, inputs, self.get_sandbox())
        return self.execute_without_sandbox(tool.id, inputs)

    def execute_with_sandbox(
        self,
        tool_id: str,
        inputs: dict,
        sandbox: Sandbox,
    ) -> ToolExecutionResult:
        """
        Execute a tool with a sandbox.
        Args:
            tool_id: The id of the tool.
            inputs: The inputs to the tool.
            sandbox: The sandbox to use.
        Returns:
            The result of the tool execution.
        """
        sandbox.execute(inputs)
        return self.execute_without_sandbox(tool_id, inputs)
