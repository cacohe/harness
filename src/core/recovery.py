from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from src.core.execution import LifecycleError, LifecycleTracker
from src.core.identifiers import ExecutionIdFactory, UuidExecutionIds
from src.core.persistence import ExecutionRecord, ExecutionStore
from src.schema.execution import (
    PendingSessionEvent,
    SessionEvent,
    SessionEventType,
    SubmissionRecord,
    SubmissionState,
)
from src.schema.task import TaskFailReason, TaskState


class RecoveryError(RuntimeError):
    """Durable history cannot be recovered without guessing or replaying side effects."""


@dataclass(frozen=True)
class _OpenLifecycles:
    submissions: list[SessionEvent]
    executions: list[SessionEvent]
    turns: list[SessionEvent]
    steps: list[SessionEvent]
    attempts: list[SessionEvent]
    tools: list[SessionEvent]
    started_tool_events: set[str]


class ExecutionRecovery:
    """Validate one session and close interrupted lifecycles without replaying work."""

    def __init__(
        self,
        store: ExecutionStore,
        id_factory: ExecutionIdFactory | None = None,
    ) -> None:
        self.store = store
        self.id_factory = id_factory if id_factory is not None else UuidExecutionIds()

    def recover_session(self, session_id: str) -> list[SessionEvent]:
        history = self.store.events(session_id)
        self._validate_sequence(history)
        try:
            LifecycleTracker().extend(history)
        except LifecycleError as error:
            raise RecoveryError("session lifecycle history is invalid") from error

        records = self.store.unfinished(session_id)
        for record in records:
            self._validate_task_identity(record, history)
        recovered: list[SessionEvent] = []
        for record in records:
            if record.task.state is TaskState.TASK_PAUSED:
                continue
            recovered.extend(self._recover_record(record, history))
            history = self.store.events(session_id)
        return recovered

    @staticmethod
    def _validate_sequence(events: Sequence[SessionEvent]) -> None:
        for expected, event in enumerate(events, start=1):
            if event.sequence != expected:
                raise RecoveryError("session event sequence contains a gap")

    @staticmethod
    def _validate_task_identity(
        record: ExecutionRecord, events: Sequence[SessionEvent]
    ) -> None:
        task_events = [event for event in events if event.task_id == record.task.id]
        if not task_events:
            raise RecoveryError("unfinished Task has no durable events")
        if record.task.session_id != record.session_id:
            raise RecoveryError("Task session identity disagrees with storage")
        if record.task.turn_id is not None and any(
            event.turn_id not in {None, record.task.turn_id} for event in task_events
        ):
            raise RecoveryError("Task turn identity disagrees with durable events")
        if record.last_sequence != events[-1].sequence:
            raise RecoveryError("Task sequence disagrees with durable events")

    def _recover_record(
        self, record: ExecutionRecord, history: Sequence[SessionEvent]
    ) -> list[SessionEvent]:
        task = record.task
        events = [event for event in history if event.task_id == task.id]
        open_items = self._open_lifecycles(events)
        last = events[-1]
        pending: list[PendingSessionEvent] = []

        for event in reversed(open_items.tools):
            started = event.event_id in open_items.started_tool_events
            pending.append(
                self._settlement(
                    last,
                    SessionEventType.TOOL_CALL_SETTLED,
                    turn_id=event.turn_id,
                    step_id=event.step_id,
                    tool_call_id=event.tool_call_id,
                    outcome="interrupted" if started else "rejected",
                    extra={"result": "unknown" if started else "not_started"},
                )
            )
        for event in reversed(open_items.attempts):
            pending.append(
                self._settlement(
                    last,
                    SessionEventType.MODEL_ATTEMPT_SETTLED,
                    turn_id=event.turn_id,
                    step_id=event.step_id,
                    attempt_id=event.attempt_id,
                    outcome="interrupted",
                )
            )
        for event in reversed(open_items.steps):
            pending.append(
                self._settlement(
                    last,
                    SessionEventType.STEP_SETTLED,
                    turn_id=event.turn_id,
                    step_id=event.step_id,
                    outcome="interrupted",
                )
            )
        for event in reversed(open_items.turns):
            pending.append(
                self._settlement(
                    last,
                    SessionEventType.TURN_SETTLED,
                    turn_id=event.turn_id,
                    outcome="interrupted",
                )
            )
        for event in reversed(open_items.submissions):
            pending.append(
                self._settlement(
                    last,
                    SessionEventType.SUBMISSION_SETTLED,
                    turn_id=event.turn_id,
                    outcome="interrupted",
                )
            )
        for event in reversed(open_items.executions):
            pending.append(
                self._settlement(
                    event,
                    SessionEventType.EXECUTION_SETTLED,
                    turn_id=event.turn_id,
                    outcome="interrupted",
                )
            )

        if not pending:
            raise RecoveryError("unfinished Task has no recoverable open lifecycle")
        task.transition_to_failed(TaskFailReason.TASK_FAILED_INTERRUPTED)
        submission = self._interrupted_submission(task)
        try:
            LifecycleTracker().extend([*history, *pending])
        except LifecycleError as error:
            raise RecoveryError("interruption compensation is invalid") from error
        return self.store.commit(
            task,
            pending,
            expected_sequence=record.last_sequence,
            submission=submission,
        )

    def _interrupted_submission(self, task) -> SubmissionRecord | None:
        if task.message_id is None:
            return None
        existing = self.store.submission(task.session_id, task.message_id)
        if existing is None:
            return None
        return SubmissionRecord.model_validate(
            {
                **existing.model_dump(),
                "state": SubmissionState.INTERRUPTED,
                "updated_at": datetime.now(timezone.utc),
            }
        )

    def _settlement(
        self,
        source: SessionEvent,
        event_type: SessionEventType,
        *,
        outcome: str,
        turn_id: str | None,
        step_id: str | None = None,
        attempt_id: str | None = None,
        tool_call_id: str | None = None,
        extra: dict | None = None,
    ) -> PendingSessionEvent:
        return PendingSessionEvent(
            event_id=self.id_factory.event(),
            session_id=source.session_id,
            type=event_type,
            task_id=source.task_id,
            execution_id=source.execution_id,
            turn_id=turn_id,
            step_id=step_id,
            attempt_id=attempt_id,
            tool_call_id=tool_call_id,
            payload={"outcome": outcome, **(extra or {})},
        )

    @staticmethod
    def _open_lifecycles(events: Sequence[SessionEvent]) -> _OpenLifecycles:
        starts = {
            "submissions": (SessionEventType.SUBMISSION_ACCEPTED, SessionEventType.SUBMISSION_SETTLED, "turn_id"),
            "executions": (SessionEventType.EXECUTION_STARTED, SessionEventType.EXECUTION_SETTLED, "execution_id"),
            "turns": (SessionEventType.TURN_STARTED, SessionEventType.TURN_SETTLED, "turn_id"),
            "steps": (SessionEventType.STEP_STARTED, SessionEventType.STEP_SETTLED, "step_id"),
            "attempts": (SessionEventType.MODEL_ATTEMPT_STARTED, SessionEventType.MODEL_ATTEMPT_SETTLED, "attempt_id"),
        }
        result: dict[str, list[SessionEvent]] = {}
        for name, (started_type, settled_type, field) in starts.items():
            opened: dict[str, SessionEvent] = {}
            for event in events:
                identity = getattr(event, field)
                if identity is None:
                    continue
                if event.type is started_type:
                    opened[identity] = event
                elif event.type is settled_type:
                    opened.pop(identity, None)
            result[name] = list(opened.values())

        requested: dict[str, SessionEvent] = {}
        started_tools: set[str] = set()
        started_tool_events: set[str] = set()
        for event in events:
            identity = event.tool_call_id
            if identity is None:
                continue
            if event.type is SessionEventType.TOOL_CALL_REQUESTED:
                requested[identity] = event
            elif event.type is SessionEventType.TOOL_CALL_STARTED:
                started_tools.add(identity)
                if identity in requested:
                    started_tool_events.add(requested[identity].event_id)
            elif event.type is SessionEventType.TOOL_CALL_SETTLED:
                requested.pop(identity, None)
                started_tools.discard(identity)
        return _OpenLifecycles(
            submissions=result["submissions"],
            executions=result["executions"],
            turns=result["turns"],
            steps=result["steps"],
            attempts=result["attempts"],
            tools=list(requested.values()),
            started_tool_events=started_tool_events,
        )
