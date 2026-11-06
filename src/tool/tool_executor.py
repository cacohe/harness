from src.core.sandbox import Sandbox
from src.core.tool import ToolExecutor, ToolRegistry, ToolResult


class DefaultToolExecutor(ToolExecutor):
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def is_need_approval(self, tool_id: str) -> bool:
        """
        Check if the tool needs approval.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the tool needs approval, False otherwise.
        """
        ...

    def launch_approval(self, tool_id: str) -> bool:
        """
        Launch the approval process.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the approval process is launched, False otherwise.
        """
        ...

    def approve(self, tool_id: str) -> bool:
        """
        Approve a tool.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the tool is approved, False otherwise.
        """
        ...

    def is_need_sandbox(self, tool_id: str) -> bool:
        """
        Check if the tool needs sandbox.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the tool needs sandbox, False otherwise.
        """
        ...

    def get_sandbox(self, tool_id: str) -> Sandbox:
        """
        Get the sandbox for a tool.
        Args:
            tool_id: The id of the tool.
        Returns:
            The sandbox for the tool.
        """
        ...

    def execute(self, tool_id: str, inputs: dict) -> ToolResult:
        """
        Execute a tool.
        Args:
            tool: The tool to execute.
            inputs: The inputs to the tool.
        Returns:
            The result of the tool execution.
        """
        is_need_approval = self.is_need_approval(tool_id)
        is_need_sandbox = self.is_need_sandbox(tool_id)
        if is_need_approval:
            self.launch_approval(tool_id)

        if is_need_sandbox:
            sandbox = self.get_sandbox(tool_id)
            return self.execute_with_sandbox(tool_id, inputs, sandbox)

        return self.tool_registry.get_tool(tool_id).executer(inputs)

    def execute_with_sandbox(
        self,
        tool_id: str,
        inputs: dict,
    ) -> ToolResult:
        """
        Execute a tool with a sandbox.
        Args:
            tool_id: The id of the tool.
            inputs: The inputs to the tool.
            sandbox: The sandbox to use.
        Returns:
            The result of the tool execution.
        """
        sandbox = self.get_sandbox(tool_id)
        sandbox.execute(inputs)
        return self.tool_registry.get_tool(tool_id).executer(inputs)
