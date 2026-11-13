from src.core.context import build_tool_messages
from src.schema.tool import (
    ToolCallResult,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolExecutionState,
)


def _result(tool_id, state, *, reason=None):
    return ToolExecutionResult(
        tool_id=tool_id,
        tool_name=tool_id,
        state=state,
        fail_reason=reason,
    )


def test_build_tool_messages_prefers_ordered_tools():
    first = _result("call-1", ToolExecutionState.TOOL_EXECUTION_STATE_PENDING)
    second = _result(
        "call-2",
        ToolExecutionState.TOOL_EXECUTION_STATE_FAILED,
        reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED,
    )
    third = _result("call-3", ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED)
    result = ToolCallResult(
        pending_tools=[first],
        completed_tools=[third],
        failed_tools=[second],
        ordered_tools=[first, second, third],
    )

    messages = build_tool_messages(result)

    assert [message.tool_call_id for message in messages] == ["call-1", "call-2", "call-3"]


def test_build_tool_messages_falls_back_for_legacy_results():
    completed = _result("completed", ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED)
    failed = _result("failed", ToolExecutionState.TOOL_EXECUTION_STATE_FAILED)
    pending = _result("pending", ToolExecutionState.TOOL_EXECUTION_STATE_PENDING)
    result = ToolCallResult(
        completed_tools=[completed],
        failed_tools=[failed],
        pending_tools=[pending],
    )

    messages = build_tool_messages(result)

    assert [message.tool_call_id for message in messages] == ["completed", "failed", "pending"]
