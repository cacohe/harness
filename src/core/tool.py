import json
from abc import ABC, abstractmethod

from src.core.sandbox import Sandbox
from src.schema.llm import LlmToolCall
from src.schema.tool import (
    ApprovalDecision,
    Tool,
    ToolCallResult,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)


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

    def execute_tool_calls(
        self, tool_calls: list[LlmToolCall], task_id: str | None = None
    ) -> ToolCallResult:
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
            tool_execution_result = self.execute_tool_call(call, hold=bool(pending_tools))
            if tool_execution_result.is_completed():
                completed_tools.append(tool_execution_result)
            elif tool_execution_result.is_failed():
                failed_tools.append(tool_execution_result)
            else:
                pending_tools.append(tool_execution_result)

        return ToolCallResult(
            pending_tools=pending_tools,
            completed_tools=completed_tools,
            failed_tools=failed_tools,
        )

    def execute_tool_call(
        self, call: LlmToolCall, approved: bool = False, hold: bool = False
    ) -> ToolExecutionResult:
        """
        Run one model tool call.
        Args:
            call: The tool call from the model. It has id, name, and arguments.
            approved: True when the user already approved this call.
            hold: True when an earlier call in the same batch waits for approval.
        Returns:
            The execution result for that call. A tool that needs approval, or any call after
            one, is held as pending with the model arguments, unless the call is approved.
        """
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
        needs_approval = not approved and self.is_need_approval(tool)
        if needs_approval or hold:
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
                arguments=call.arguments,
                needs_approval=needs_approval,
            )
        result = self.execute(tool, parsed)
        result.tool_id = call.id
        return result

    def execute_held_tool_call(self, held: ToolExecutionResult) -> ToolExecutionResult:
        """
        Run or reject a held tool call after the user decided.
        Args:
            held: The pending result. Its approval is the user's decision when it needs one.
        Returns:
            The execution result when the call can run. Otherwise a failed result with the reason.
        """
        call = LlmToolCall(held.tool_id, held.tool_name, held.arguments or "")
        decision = held.approval
        if held.can_run():
            return self.execute_tool_call(call, approved=True)
        if decision == ApprovalDecision.APPROVAL_DECISION_TIMEOUT:
            reason = ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_TIMEOUT
            output = "The approval timed out."
        else:
            reason = ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED
            output = "The tool call was rejected."
        return ToolExecutionResult(
            tool_id=call.id,
            tool_name=call.name,
            state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
            output=output,
            fail_reason=reason,
            arguments=call.arguments,
            approval=decision,
        )
