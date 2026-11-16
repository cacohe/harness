import json
import math

import pytest
from pydantic import ValidationError
from src.core.task import Task
from src.schema.context import Message
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall, LlmUsage
from src.schema.task import (
    TaskFailReason,
    TaskPauseReason,
    TaskPlan,
    TaskResult,
    TaskRunningReason,
    TaskState,
)
from src.schema.tool import (
    ApprovalDecision,
    ToolExecutionResult,
    ToolExecutionState,
)


def test_task_plan_does_not_expose_an_unimplemented_advancement_api():
    assert not hasattr(TaskPlan, "advance_plan")


def test_new_tasks_have_independent_identity_and_collections():
    first = Task(original_input="first")
    second = Task(original_input="second")

    first.history.append("message")
    first.pending_tool_calls.append("call")

    assert isinstance(first.id, str)
    assert first.id != second.id
    assert second.history == []
    assert second.pending_tool_calls == []


def test_new_task_can_be_serialized_before_inference():
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


def test_task_accepts_legacy_id_and_none_collections():
    task = Task("legacy", "known-id", history=None, pending_tool_calls=None)
    generated = Task("generated", id=None)

    assert task.id == "known-id"
    assert task.history == []
    assert task.pending_tool_calls == []
    assert generated.id


def test_failed_transition_does_not_partially_mutate_task():
    task = Task("atomic")
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)

    with pytest.raises(ValidationError):
        task.transition_to_completed()

    assert task.state is TaskState.TASK_RUNNING
    assert task.state_reason is TaskRunningReason.TASK_RUNNING_CREATE


def test_paused_task_can_transition_directly_to_cancelled_failure():
    task = Task("cancel")
    task.transition_to_running(TaskRunningReason.TASK_RUNNING_CREATE)
    task.transition_to_paused(TaskPauseReason.TASK_PAUSED_WAITING_APPROVAL)
    task.set_result(TaskResult(output="Task cancelled."))

    task.transition_to_failed(TaskFailReason.TASK_FAILED_CANCEL)

    assert task.state is TaskState.TASK_FAILED
    assert task.state_reason is TaskFailReason.TASK_FAILED_CANCEL


def test_invalid_terminal_result_assignment_preserves_previous_result():
    result = TaskResult(output="done")
    task = Task("atomic", state=TaskState.TASK_COMPLETED, result=result)

    with pytest.raises(ValidationError):
        task.result = None

    assert task.result == result


def test_invalid_task_result_assignment_preserves_previous_output():
    result = TaskResult(output="done")

    with pytest.raises(ValidationError):
        result.output = 123

    assert result.output == "done"


@pytest.mark.parametrize("field", ["id", "history", "pending_tool_calls"])
def test_none_assignment_does_not_apply_constructor_compatibility(field):
    task = Task(
        "atomic",
        id="known-id",
        history=[Message(role="user", content="keep")],
        pending_tool_calls=[
            ToolExecutionResult(
                tool_id="call-1",
                tool_name="keep",
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
            )
        ],
    )
    previous = getattr(task, field)

    with pytest.raises(ValidationError):
        setattr(task, field, None)

    assert getattr(task, field) == previous


@pytest.mark.parametrize(
    ("field", "item"),
    [
        ("history", Message(role="user", content="keep")),
        (
            "pending_tool_calls",
            ToolExecutionResult(
                tool_id="call-1",
                tool_name="keep",
                state=ToolExecutionState.TOOL_EXECUTION_STATE_PENDING,
            ),
        ),
    ],
)
def test_iterable_assignment_commits_the_validated_items(field, item):
    task = Task("atomic")

    setattr(task, field, iter([item]))

    assert getattr(task, field) == [item]


def test_state_reason_assignment_uses_the_current_task_state():
    task = Task(
        "failed",
        state=TaskState.TASK_FAILED,
        state_reason=TaskFailReason.TASK_FAILED_CANCEL,
        result=TaskResult(output="cancelled"),
    )

    task.state_reason = "unknown"

    assert task.state_reason is TaskFailReason.TASK_FAILED_UNKNOWN


@pytest.mark.parametrize(
    ("field", "value"),
    [("step_count", -1), ("elapsed_seconds", -0.01)],
)
def test_task_rejects_negative_progress(field, value):
    with pytest.raises(ValidationError):
        Task(original_input="invalid", **{field: value})


def test_task_rejects_non_finite_elapsed_time():
    with pytest.raises(ValidationError):
        Task(original_input="invalid", elapsed_seconds=math.inf)


@pytest.mark.parametrize("output", [math.nan, math.inf, {1: "integer", "1": "string"}])
def test_task_snapshot_export_rejects_lossy_json_values(output):
    task = Task(
        "invalid snapshot",
        pending_tool_calls=[
            ToolExecutionResult(
                tool_id="call-1",
                tool_name="lossy",
                state=ToolExecutionState.TOOL_EXECUTION_STATE_COMPLETED,
                output=output,
            )
        ],
    )

    with pytest.raises(ValueError):
        task.to_dict()


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


def test_legacy_terminal_snapshot_discards_stale_running_reason():
    task = Task.from_dict(
        {
            "id": "legacy-id",
            "original_input": "done",
            "state": "completed",
            "state_reason": "create",
            "result": {"output": "done"},
            "history": None,
            "pending_tool_calls": None,
        }
    )

    assert task.schema_version == 1
    assert task.state is TaskState.TASK_COMPLETED
    assert task.state_reason is None
    assert task.history == []
    assert task.pending_tool_calls == []
