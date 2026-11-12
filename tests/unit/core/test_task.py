from src.core.task import Task
from src.schema.context import Message
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall, LlmUsage
from src.schema.tool import ApprovalDecision, ToolExecutionResult, ToolExecutionState


def test_new_tasks_have_independent_identity_and_collections():
    first = Task(original_input="first")
    second = Task(original_input="second")

    first.history.append("message")
    first.pending_tool_calls.append("call")

    assert isinstance(first.id, str)
    assert first.id != second.id
    assert second.history == []
    assert second.pending_tool_calls == []


def test_new_task_can_be_serialized_before_planning():
    task = Task(original_input="inspect project")

    serialized = task.to_dict()

    assert serialized["plan"] is None
    assert serialized["result"] is None


def test_nested_task_state_models_round_trip_through_json_data():
    call = LlmToolCall(id="call-1", name="search", arguments='{"query":"phone"}')
    message = Message(
        role="assistant",
        content=None,
        tool_calls=[call],
        reasoning_content="compare specifications",
        thinking_blocks=[{"type": "thinking", "thinking": "compare"}],
    )
    output = LLMOutput(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=[call],
        reasoning_content="compare specifications",
        thinking_blocks=None,
        usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    pending = ToolExecutionResult(
        tool_id="call-1",
        tool_name="search",
        state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
        arguments='{"query":"phone"}',
        approval=ApprovalDecision.APPROVAL_DECISION_APPROVED,
        needs_approval=True,
    )

    message_data = message.model_dump(mode="json")
    output_data = output.model_dump(mode="json")
    pending_data = pending.model_dump(mode="json")

    assert message_data["tool_calls"] == [
        {"id": "call-1", "name": "search", "arguments": '{"query":"phone"}'}
    ]
    assert output_data["finish_reason"] == "tool_calls"
    assert output_data["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert pending_data["state"] == "pending"
    assert pending_data["approval"] == "approved"
    assert Message.model_validate(message_data) == message
    assert LLMOutput.model_validate(output_data) == output
    assert ToolExecutionResult.model_validate(pending_data) == pending
