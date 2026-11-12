import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from src.schema.llm import LlmToolCall
from src.schema.tool import (
    ApprovalDecision,
    Tool,
    ToolCallResult,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)


@runtime_checkable
class ToolCatalog(Protocol):
    def register(self, tool: Tool) -> None: ...
    def unregister(self, tool_id: str) -> None: ...
    def get_tool(self, tool_id: str) -> Tool: ...
    def find_tool(self, name: str) -> Tool | None: ...
    def get_all_tools(self) -> list[Tool]: ...


@runtime_checkable
class ToolRunner(Protocol):
    def run(self, tool: Tool, inputs: dict) -> Any: ...


@runtime_checkable
class ApprovalPolicy(Protocol):
    def decide(self, tool: Tool, inputs: dict) -> ApprovalDecision | None: ...


@runtime_checkable
class SandboxRunner(Protocol):
    def supports(self, tool: Tool) -> bool: ...
    def run(self, tool: Tool, inputs: dict) -> Any: ...


class AllowAllApprovalPolicy:
    """Approve every tool call."""

    def decide(self, tool: Tool, inputs: dict) -> ApprovalDecision:
        return ApprovalDecision.APPROVAL_DECISION_APPROVED


class ToolExecutor:
    """Parse, authorize, route, and classify model tool calls."""

    def __init__(
        self,
        catalog: ToolCatalog,
        runner: ToolRunner,
        *,
        approval_policy: ApprovalPolicy | None = None,
        sandbox_runner: SandboxRunner | None = None,
    ) -> None:
        self.catalog = catalog
        self.runner = runner
        self.approval_policy = (
            approval_policy if approval_policy is not None else AllowAllApprovalPolicy()
        )
        self.sandbox_runner = sandbox_runner

    def get_all_tools(self) -> list[Tool]:
        return self.catalog.get_all_tools()

    def find_tool(self, name: str) -> Tool | None:
        return self.catalog.find_tool(name)

    def execute(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        try:
            decision = self.approval_policy.decide(tool, inputs)
        except Exception as error:
            return self._failed(tool, output=f"{type(error).__name__}: {error}")
        if decision is None:
            return ToolExecutionResult(
                tool_id=tool.id,
                tool_name=tool.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
            )
        if decision is ApprovalDecision.APPROVAL_DECISION_REJECTED:
            return self._failed(tool, ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REJECTED)
        if decision is ApprovalDecision.APPROVAL_DECISION_TIMEOUT:
            return self._failed(tool, ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_TIMEOUT)
        if decision is not ApprovalDecision.APPROVAL_DECISION_APPROVED:
            return self._failed(tool)
        return self.execute_approved(tool, inputs)

    def execute_approved(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        """Run a tool after authorization has already been established."""
        try:
            if self.sandbox_runner is not None and self.sandbox_runner.supports(tool):
                output = self.sandbox_runner.run(tool, inputs)
            else:
                output = self.runner.run(tool, inputs)
        except Exception as error:
            return self._failed(tool, output=f"{type(error).__name__}: {error}")
        return ToolExecutionResult(
            tool_id=tool.id,
            tool_name=tool.name,
            state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
            output=output,
        )

    @staticmethod
    def _failed(
        tool: Tool,
        reason: ToolExecutionFailReason = ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_UNKNOWN,
        *,
        output: Any | None = None,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_id=tool.id,
            tool_name=tool.name,
            state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
            output=output,
            fail_reason=reason,
        )

    def execute_tool_calls(
        self,
        tool_calls: list[LlmToolCall],
        *,
        before_call: Callable[[LlmToolCall], None] | None = None,
        after_call: Callable[[LlmToolCall, ToolExecutionResult], None] | None = None,
    ) -> ToolCallResult:
        pending_tools: list[ToolExecutionResult] = []
        completed_tools: list[ToolExecutionResult] = []
        failed_tools: list[ToolExecutionResult] = []

        for call in tool_calls:
            if before_call is not None:
                before_call(call)
            result = self.execute_tool_call(call, hold=bool(pending_tools))
            if after_call is not None:
                after_call(call, result)
            if result.is_completed():
                completed_tools.append(result)
            elif result.is_failed():
                failed_tools.append(result)
            else:
                pending_tools.append(result)

        return ToolCallResult(
            pending_tools=pending_tools,
            completed_tools=completed_tools,
            failed_tools=failed_tools,
        )

    def execute_tool_call(
        self, call: LlmToolCall, approved: bool = False, hold: bool = False
    ) -> ToolExecutionResult:
        """Run one model tool call, or hold it until an earlier approval is resolved."""
        try:
            parsed = json.loads(call.arguments) if call.arguments else {}
        except (json.JSONDecodeError, TypeError):
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
                fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_NOT_SUPPORTED,
            )
        if hold:
            try:
                decision = self.approval_policy.decide(tool, parsed)
            except Exception as error:
                result = self._failed(tool, output=f"{type(error).__name__}: {error}")
                result.tool_id = call.id
                return result
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
                arguments=call.arguments,
                approval=decision if decision is not None else None,
                needs_approval=decision is None,
            )
        result = self.execute_approved(tool, parsed) if approved else self.execute(tool, parsed)
        result.tool_id = call.id
        if result.is_pending():
            result.arguments = call.arguments
            result.needs_approval = True
        return result

    def execute_held_tool_call(self, held: ToolExecutionResult) -> ToolExecutionResult:
        """Run or reject a held tool call after the user decides."""
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
