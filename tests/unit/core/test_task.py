import json

import pytest
from pydantic import ValidationError
from src.core.task import Task
from src.schema.context import Message
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall, LlmUsage
from src.schema.task import TaskFailReason, TaskPlan, TaskResult, TaskState
from src.schema.tool import (
    ApprovalDecision,
    ToolExecutionResult,
    ToolExecutionState,
)


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


def test_task_round_trips_all_snapshot_fields_through_json_data():
    task = Task(
        "inspect project",
        state=TaskState.TASK_FAILED,
        state_reason=TaskFailReason.TASK_FAILED_UNKNOWN,
        plan=TaskPlan(goal=["inspect"], current="report"),
        history=[Message(role="assistant", content="working")],
        pending_tool_calls=[
            ToolExecutionResult(
                tool_id="call-1",
                tool_name="read_file",
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
                arguments='{"path":"README.md"}',
            )
        ],
        result=TaskResult(output="could not finish"),
        step_count=3,
        elapsed_seconds=1.5,
    )

    serialized = task.to_dict()
    encoded = json.dumps(serialized)
    restored = Task.from_dict(json.loads(encoded))

    assert serialized["schema_version"] == 1
    assert restored == task
    assert restored is not task
    assert restored.history[0] is not task.history[0]


def test_task_accepts_positional_and_keyword_original_input():
    assert Task("positional").original_input == "positional"
    assert Task(original_input="keyword").original_input == "keyword"


@pytest.mark.parametrize(
    ("field", "value"),
    [("step_count", -1), ("elapsed_seconds", -0.01)],
)
def test_task_rejects_negative_progress(field, value):
    with pytest.raises(ValidationError):
        Task(original_input="invalid", **{field: value})


@pytest.mark.parametrize("state", [TaskState.TASK_COMPLETED, TaskState.TASK_FAILED])
def test_terminal_task_requires_a_result(state):
    with pytest.raises(ValidationError):
        Task(original_input="invalid", state=state)


def test_failed_unknown_reason_restores_as_fail_reason():
    task = Task.from_dict(
        {
            "original_input": "failed",
            "state": "failed",
            "state_reason": "unknown",
            "result": {"output": "failed"},
        }
    )

    assert task.state_reason is TaskFailReason.TASK_FAILED_UNKNOWN
