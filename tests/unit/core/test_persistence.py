from src.core import persistence
from src.core.task import Task
from src.schema.execution import PendingSessionEvent, SessionEventType


class RecordingTaskRepository:
    def __init__(self) -> None:
        self.task = None

    def save(self, task):
        self.task = Task.from_dict(task.to_dict())

    def get(self, task_id):
        if self.task is None or self.task.id != task_id:
            return None
        return Task.from_dict(self.task.to_dict())


def test_core_exposes_task_repository_port():
    assert hasattr(persistence, "TaskRepository")


def test_snapshot_execution_store_accepts_none_sequence_for_legacy_repository():
    legacy = RecordingTaskRepository()
    store = persistence.SnapshotExecutionStore(legacy)
    task = Task("legacy", id="task-1", session_id="session-1")
    event = PendingSessionEvent(
        event_id="event-1",
        session_id="session-1",
        type=SessionEventType.TURN_STARTED,
        task_id="task-1",
        execution_id="execution-1",
        turn_id="turn-1",
        payload={},
    )

    committed = store.commit(task, [event], expected_sequence=None)

    assert [item.sequence for item in committed] == [1]
    assert legacy.get("task-1").last_sequence == 1
    assert store.events("session-1") == committed
