from src.core.task import Task


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
