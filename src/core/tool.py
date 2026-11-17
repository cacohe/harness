import asyncio
import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from src.core.async_utils import call_maybe_async, run_sync
from src.core.workspace import Workspace, WorkspaceViolation
from src.schema.llm import LlmToolCall
from src.schema.tool import (
    TOOL_CALL_CANCELLED_OUTPUT,
    ApprovalDecision,
    Tool,
    ToolCallResult,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)


class RemoteToolError(Exception):
    """A remote tool returned a structured failure."""

    def __init__(self, output: Any):
        self.output = output
        super().__init__(str(output))


def _public_tool_error(error: BaseException, operation: str) -> str:
    public_message = getattr(error, "public_message", None)
    if isinstance(public_message, str) and public_message:
        return public_message[:1000]
    return f"{type(error).__name__}: {operation} failed"


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
class AsyncToolRunner(Protocol):
    async def arun(self, tool: Tool, inputs: dict) -> Any: ...


@runtime_checkable
class ApprovalPolicy(Protocol):
    def decide(self, tool: Tool, inputs: dict) -> ApprovalDecision | None: ...


@runtime_checkable
class SandboxRunner(Protocol):
    def supports(self, tool: Tool) -> bool: ...
    def run(self, tool: Tool, inputs: dict, workspace: Workspace) -> Any: ...


class AllowAllApprovalPolicy:
    """Approve every tool call."""

    def decide(self, tool: Tool, inputs: dict) -> ApprovalDecision:
        return ApprovalDecision.APPROVAL_DECISION_APPROVED


class ToolExecutor:
    """Parse, authorize, route, and classify model tool calls."""

    def __init__(
        self,
        catalog: ToolCatalog,
        runner: ToolRunner | AsyncToolRunner,
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
        return run_sync(lambda: self.aexecute(tool, inputs), async_name="aexecute")

    async def aexecute(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        validated, failure = self._validate_inputs(tool, inputs)
        if failure is not None:
            return failure
        assert validated is not None
        return await self._aexecute(tool, validated)

    def _execute(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        return run_sync(lambda: self._aexecute(tool, inputs), async_name="aexecute")

    async def _aexecute(
        self,
        tool: Tool,
        inputs: dict,
        *,
        before_execute: Callable[[], None] | None = None,
    ) -> ToolExecutionResult:
        try:
            decision = self.approval_policy.decide(tool, inputs)
        except Exception as error:
            return self._failed(tool, output=_public_tool_error(error, "approval"))
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
        if before_execute is not None:
            before_execute()
        return await self._arun_approved(tool, inputs)

    def execute_approved(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        return run_sync(
            lambda: self.aexecute_approved(tool, inputs), async_name="aexecute_approved"
        )

    async def aexecute_approved(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        """Run a tool after authorization has already been established."""
        validated, failure = self._validate_inputs(tool, inputs)
        if failure is not None:
            return failure
        assert validated is not None
        return await self._arun_approved(tool, validated)

    def _run_approved(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        return run_sync(lambda: self._arun_approved(tool, inputs), async_name="aexecute_approved")

    async def _arun_approved(self, tool: Tool, inputs: dict) -> ToolExecutionResult:
        if tool.requires_sandbox:
            if tool.workspace is None:
                return self._failed(
                    tool,
                    ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION,
                    output="WorkspaceViolation: sandbox-required tool has no bound workspace",
                )
        try:
            if tool.requires_sandbox:
                if self.sandbox_runner is None or not self.sandbox_runner.supports(tool):
                    return self._failed(
                        tool,
                        ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_SANDBOX_REQUIRED,
                        output="SandboxRequired: no compatible sandbox runner is available",
                    )
                assert self.sandbox_runner is not None
                assert tool.workspace is not None
                output = await call_maybe_async(
                    getattr(self.sandbox_runner, "arun", None),
                    getattr(self.sandbox_runner, "run", None),
                    tool,
                    inputs,
                    tool.workspace,
                )
            else:
                output = await call_maybe_async(
                    getattr(self.runner, "arun", None),
                    getattr(self.runner, "run", None),
                    tool,
                    inputs,
                )
        except TimeoutError:
            return self._failed(
                tool,
                ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_TIMEOUT,
                output="TimeoutError: tool execution timed out",
            )
        except RemoteToolError:
            return self._failed(
                tool,
                ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REMOTE_ERROR,
                output="RemoteToolError: remote tool failed",
            )
        except Exception as error:
            return self._failed(tool, output=_public_tool_error(error, "tool execution"))
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
        before_execute: Callable[[LlmToolCall], None] | None = None,
        after_call: Callable[[LlmToolCall, ToolExecutionResult], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> ToolCallResult:
        return run_sync(
            lambda: self.aexecute_tool_calls(
                tool_calls,
                before_call=before_call,
                before_execute=before_execute,
                after_call=after_call,
                should_cancel=should_cancel,
                timeout_seconds=timeout_seconds,
            ),
            async_name="aexecute_tool_calls",
        )

    async def aexecute_tool_calls(
        self,
        tool_calls: list[LlmToolCall],
        *,
        before_call: Callable[[LlmToolCall], None] | None = None,
        before_execute: Callable[[LlmToolCall], None] | None = None,
        after_call: Callable[[LlmToolCall, ToolExecutionResult], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        timeout_seconds: float | None = None,
    ) -> ToolCallResult:
        pending_tools: list[ToolExecutionResult] = []
        completed_tools: list[ToolExecutionResult] = []
        failed_tools: list[ToolExecutionResult] = []
        ordered_tools: list[ToolExecutionResult] = []

        for call in tool_calls:
            cancelled = should_cancel is not None and should_cancel()
            if not cancelled and before_call is not None:
                before_call(call)
                cancelled = should_cancel is not None and should_cancel()
            if cancelled:
                result = ToolExecutionResult(
                    tool_id=call.id,
                    tool_name=call.name,
                    state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                    output=TOOL_CALL_CANCELLED_OUTPUT,
                    fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED,
                )
                if after_call is not None:
                    after_call(call, result)
                failed_tools.append(result)
                ordered_tools.append(result)
                continue
            try:
                if timeout_seconds is None:
                    result = await self._aexecute_tool_call_compat(
                        call,
                        hold=bool(pending_tools),
                        before_execute=before_execute,
                    )
                else:
                    async with asyncio.timeout(timeout_seconds):
                        result = await self._aexecute_tool_call_compat(
                            call,
                            hold=bool(pending_tools),
                            before_execute=before_execute,
                        )
            except TimeoutError:
                result = ToolExecutionResult(
                    tool_id=call.id,
                    tool_name=call.name,
                    state=ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
                    output="TimeoutError: tool call timed out",
                    fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_TIMEOUT,
                )
            if after_call is not None:
                after_call(call, result)
            if result.is_completed():
                completed_tools.append(result)
            elif result.is_failed():
                failed_tools.append(result)
            else:
                pending_tools.append(result)
            ordered_tools.append(result)

        return ToolCallResult(
            pending_tools=pending_tools,
            completed_tools=completed_tools,
            failed_tools=failed_tools,
            ordered_tools=ordered_tools,
        )

    async def _aexecute_tool_call_compat(
        self,
        call: LlmToolCall,
        *,
        approved: bool = False,
        hold: bool = False,
        before_execute: Callable[[LlmToolCall], None] | None = None,
    ) -> ToolExecutionResult:
        sync_override = type(self).execute_tool_call is not ToolExecutor.execute_tool_call
        async_override = type(self).aexecute_tool_call is not ToolExecutor.aexecute_tool_call
        if sync_override and not async_override:
            if before_execute is not None:
                before_execute(call)
            return await call_maybe_async(
                None, self.execute_tool_call, call, approved=approved, hold=hold
            )
        if async_override:
            if before_execute is not None:
                before_execute(call)
            return await self.aexecute_tool_call(call, approved=approved, hold=hold)
        return await self.aexecute_tool_call(
            call,
            approved=approved,
            hold=hold,
            before_execute=before_execute,
        )

    def execute_tool_call(
        self, call: LlmToolCall, approved: bool = False, hold: bool = False
    ) -> ToolExecutionResult:
        return run_sync(
            lambda: self.aexecute_tool_call(call, approved=approved, hold=hold),
            async_name="aexecute_tool_call",
        )

    async def aexecute_tool_call(
        self,
        call: LlmToolCall,
        approved: bool = False,
        hold: bool = False,
        *,
        before_execute: Callable[[LlmToolCall], None] | None = None,
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
        validated, failure = self._validate_inputs(tool, parsed)
        if failure is not None:
            failure.tool_id = call.id
            return failure
        assert validated is not None
        if hold:
            try:
                decision = self.approval_policy.decide(tool, validated)
            except Exception as error:
                result = self._failed(tool, output=_public_tool_error(error, "tool execution"))
                result.tool_id = call.id
                return result
            return ToolExecutionResult(
                tool_id=call.id,
                tool_name=call.name,
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
                arguments=self._encoded_arguments(validated),
                approval=decision if decision is not None else None,
                needs_approval=decision is None,
                workspace_id=self._workspace_id(tool),
                approval_scope=tool.approval_scope,
            )
        if approved:
            if before_execute is not None:
                before_execute(call)
            result = await self._arun_approved(tool, validated)
        else:
            callback = None if before_execute is None else lambda: before_execute(call)
            result = await self._aexecute(tool, validated, before_execute=callback)
        result.tool_id = call.id
        if result.is_pending():
            result.arguments = self._encoded_arguments(validated)
            result.needs_approval = True
            result.workspace_id = self._workspace_id(tool)
            result.approval_scope = tool.approval_scope
        return result

    def execute_held_tool_call(
        self,
        held: ToolExecutionResult,
        *,
        before_execute: Callable[[LlmToolCall], None] | None = None,
    ) -> ToolExecutionResult:
        return run_sync(
            lambda: self.aexecute_held_tool_call(
                held,
                before_execute=before_execute,
            ),
            async_name="aexecute_held_tool_call",
        )

    async def aexecute_held_tool_call(
        self,
        held: ToolExecutionResult,
        *,
        before_execute: Callable[[LlmToolCall], None] | None = None,
    ) -> ToolExecutionResult:
        """Run or reject a held tool call after the user decides."""
        call = LlmToolCall(held.tool_id, held.tool_name, held.arguments or "")
        decision = held.approval
        if held.can_run():
            tool = self.find_tool(call.name)
            if tool is not None and held.approval_scope != tool.approval_scope:
                result = self._failed(
                    tool,
                    ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_APPROVAL_STALE,
                    output="ApprovalStale: approval belongs to a different tool generation",
                )
                result.tool_id = call.id
                return result
            if tool is not None and held.workspace_id != self._workspace_id(tool):
                result = self._failed(
                    tool,
                    ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION,
                    output="WorkspaceViolation: approval belongs to a different workspace",
                )
                result.tool_id = call.id
                return result
            return await self._aexecute_tool_call_compat(
                call,
                approved=True,
                before_execute=before_execute,
            )
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

    def _validate_inputs(
        self, tool: Tool, inputs: dict
    ) -> tuple[dict | None, ToolExecutionResult | None]:
        try:
            validated = dict(inputs)
            if tool.input_validator is not None:
                validated = tool.input_validator(validated)
            return validated, None
        except WorkspaceViolation as error:
            return None, self._failed(
                tool,
                ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_WORKSPACE_VIOLATION,
                output=f"WorkspaceViolation: {error}",
            )
        except Exception as error:
            return None, self._failed(tool, output=_public_tool_error(error, "input validation"))

    @staticmethod
    def _workspace_id(tool: Tool) -> str | None:
        return tool.workspace.identity if tool.workspace is not None else None

    @staticmethod
    def _encoded_arguments(inputs: dict) -> str:
        return json.dumps(inputs, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
