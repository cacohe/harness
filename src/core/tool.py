from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from src.core.sandbox import Sandbox


@dataclass
class Tool:
    id: str
    name: str
    description: str
    input_schema: dict
    executer: Callable


@dataclass
class ToolResult:
    is_success: bool
    content: str | None
    error: str | None


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
    def is_need_approval(self, tool_id: str) -> bool:
        """
        Check if the tool needs approval.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the tool needs approval, False otherwise.
        """
        ...

    @abstractmethod
    def launch_approval(self, tool_id: str) -> bool:
        """
        Launch the approval process.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the approval process is launched, False otherwise.
        """
        ...

    @abstractmethod
    def approve(self, tool_id: str) -> bool:
        """
        Approve a tool.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the tool is approved, False otherwise.
        """
        ...

    @abstractmethod
    def is_need_sandbox(self, tool_id: str) -> bool:
        """
        Check if the tool needs sandbox.
        Args:
            tool_id: The id of the tool.
        Returns:
            True if the tool needs sandbox, False otherwise.
        """
        ...

    @abstractmethod
    def get_sandbox(self, tool_id: str) -> Sandbox:
        """
        Get the sandbox for a tool.
        Args:
            tool_id: The id of the tool.
        Returns:
            The sandbox for the tool.
        """
        ...

    @abstractmethod
    def execute(self, tool_id: str, inputs: dict) -> ToolResult:
        """
        Execute a tool.
        Args:
            tool: The tool to execute.
            inputs: The inputs to the tool.
        Returns:
            The result of the tool execution.
        """
        ...

    @abstractmethod
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
        Returns:
            The result of the tool execution.
        """
        ...
