from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from src.core.persistence import ExecutionStore
from src.core.task import Task
from src.schema.execution import (
    PendingSessionEvent,
    SessionEvent,
    SessionEventType,
    SubmissionRecord,
    validate_execution_id,
)


class LifecycleError(ValueError):
    """A durable fact would create an impossible lifecycle history."""


class PayloadLimitError(ValueError):
    """A durable event or Task snapshot exceeds its configured byte budget."""


@dataclass(frozen=True)
class ExecutionContext:
    """Stable identities and boundary controls for one Harness invocation."""

    session_id: str
    task_id: str
    execution_id: str
    turn_id: str
    message_id: str | None = None
    expected_sequence: int = 0
    deadline: float | None = None
    submission: SubmissionRecord | None = None
    capability_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "task_id", "execution_id", "turn_id"):
            validate_execution_id(getattr(self, name), name)
        if self.message_id is not None:
            validate_execution_id(self.message_id, "message_id")
        if self.capability_digest is not None:
            validate_execution_id(self.capability_digest, "capability_digest")
        if self.submission is not None:
            if self.submission.session_id != self.session_id:
                raise ValueError("submission session identity does not match execution context")
            if self.submission.task_id != self.task_id:
                raise ValueError("submission task identity does not match execution context")
            if self.submission.turn_id != self.turn_id:
                raise ValueError("submission turn identity does not match execution context")
            if self.message_id is not None and self.submission.message_id != self.message_id:
                raise ValueError("submission message identity does not match execution context")
        if (
            isinstance(self.expected_sequence, bool)
            or not isinstance(self.expected_sequence, int)
            or self.expected_sequence < 0
        ):
            raise ValueError("expected_sequence must be a non-negative integer")
        if self.deadline is not None and (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (int, float))
            or not math.isfinite(self.deadline)
        ):
            raise ValueError("deadline must be a finite number or None")


@dataclass(frozen=True)
class ExecutionLimits:
    """Byte budgets enforced before durable store mutation."""

    max_event_bytes: int = 1_048_576
    max_task_snapshot_bytes: int = 8_388_608

    def __post_init__(self) -> None:
        for name in ("max_event_bytes", "max_task_snapshot_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


_Kind = Literal["submission", "execution", "turn", "step", "attempt", "tool"]
_STARTS: dict[SessionEventType, tuple[_Kind, str]] = {
    SessionEventType.SUBMISSION_ACCEPTED: ("submission", "turn_id"),
    SessionEventType.EXECUTION_STARTED: ("execution", "execution_id"),
    SessionEventType.TURN_STARTED: ("turn", "turn_id"),
    SessionEventType.STEP_STARTED: ("step", "step_id"),
    SessionEventType.MODEL_ATTEMPT_STARTED: ("attempt", "attempt_id"),
}
_SETTLES: dict[SessionEventType, tuple[_Kind, str]] = {
    SessionEventType.SUBMISSION_SETTLED: ("submission", "turn_id"),
    SessionEventType.EXECUTION_SETTLED: ("execution", "execution_id"),
    SessionEventType.TURN_SETTLED: ("turn", "turn_id"),
    SessionEventType.STEP_SETTLED: ("step", "step_id"),
    SessionEventType.MODEL_ATTEMPT_SETTLED: ("attempt", "attempt_id"),
}
_PAUSES: dict[SessionEventType, tuple[_Kind, str]] = {
    SessionEventType.SUBMISSION_PAUSED: ("submission", "turn_id"),
    SessionEventType.TURN_PAUSED: ("turn", "turn_id"),
    SessionEventType.STEP_PAUSED: ("step", "step_id"),
}
_RESUMES: dict[SessionEventType, tuple[_Kind, str]] = {
    SessionEventType.SUBMISSION_RESUMED: ("submission", "turn_id"),
    SessionEventType.TURN_RESUMED: ("turn", "turn_id"),
    SessionEventType.STEP_RESUMED: ("step", "step_id"),
}
_TURN_CHILD_EVENTS = {
    SessionEventType.STEP_STARTED,
    SessionEventType.STEP_PAUSED,
    SessionEventType.STEP_RESUMED,
    SessionEventType.STEP_SETTLED,
    SessionEventType.MODEL_ATTEMPT_STARTED,
    SessionEventType.MODEL_ATTEMPT_SETTLED,
    SessionEventType.ASSISTANT_SETTLED,
    SessionEventType.TOOL_CALL_REQUESTED,
    SessionEventType.TOOL_CALL_STARTED,
    SessionEventType.TOOL_CALL_SETTLED,
}
_FINAL_OUTCOMES = {"completed", "failed", "cancelled", "interrupted", "rejected"}


class LifecycleTracker:
    """Purely fold durable facts while enforcing lifecycle invariants."""

    def __init__(self) -> None:
        self._started: dict[_Kind, set[str]] = {
            kind: set() for kind in ("submission", "execution", "turn", "step", "attempt", "tool")
        }
        self._settled: dict[_Kind, set[str]] = {kind: set() for kind in self._started}
        self._paused: dict[_Kind, set[str]] = {kind: set() for kind in self._started}
        self._parents: dict[tuple[_Kind, str], tuple[str | None, ...]] = {}
        self._tool_started: set[str] = set()

    def extend(self, events: Iterable[PendingSessionEvent | SessionEvent]) -> None:
        for event in events:
            self.apply(event)

    def apply(self, event: PendingSessionEvent | SessionEvent) -> None:
        if event.turn_id in self._settled["turn"] and event.type in _TURN_CHILD_EVENTS:
            raise LifecycleError("event cannot follow a terminal turn")

        if event.type in _STARTS:
            kind, field = _STARTS[event.type]
            identity = self._identity(event, field)
            if identity in self._started[kind]:
                raise LifecycleError(f"{kind} {identity} is already started")
            self._require_open_parent(event, kind)
            self._started[kind].add(identity)
            self._parents[(kind, identity)] = self._hierarchy(event, kind)
            return

        if event.type is SessionEventType.TOOL_CALL_REQUESTED:
            identity = self._identity(event, "tool_call_id")
            if identity in self._started["tool"]:
                raise LifecycleError(f"tool {identity} is already started")
            self._require_open_parent(event, "tool")
            self._started["tool"].add(identity)
            self._parents[("tool", identity)] = self._hierarchy(event, "tool")
            return

        if event.type is SessionEventType.TOOL_CALL_STARTED:
            identity = self._identity(event, "tool_call_id")
            self._require_started("tool", identity)
            self._require_hierarchy(event, "tool", identity)
            if identity in self._tool_started:
                raise LifecycleError(f"tool {identity} is already started")
            self._tool_started.add(identity)
            return

        if event.type is SessionEventType.TOOL_CALL_SETTLED:
            identity = self._identity(event, "tool_call_id")
            outcome = self._outcome(event, allow_paused=False)
            self._require_started("tool", identity)
            self._require_hierarchy(event, "tool", identity)
            if identity in self._settled["tool"]:
                raise LifecycleError(f"tool {identity} is already settled")
            if identity not in self._tool_started and outcome != "rejected":
                raise LifecycleError(f"tool {identity} was not started")
            self._settled["tool"].add(identity)
            return

        if event.type in _PAUSES:
            kind, field = _PAUSES[event.type]
            identity = self._identity(event, field)
            self._require_started(kind, identity)
            self._require_hierarchy(event, kind, identity)
            if identity in self._settled[kind]:
                raise LifecycleError(f"{kind} {identity} is already settled")
            if identity in self._paused[kind]:
                raise LifecycleError(f"{kind} {identity} is already paused")
            self._paused[kind].add(identity)
            return

        if event.type in _RESUMES:
            kind, field = _RESUMES[event.type]
            identity = self._identity(event, field)
            self._require_started(kind, identity)
            self._require_hierarchy(event, kind, identity)
            if identity not in self._paused[kind]:
                raise LifecycleError(f"{kind} {identity} is not paused")
            self._paused[kind].remove(identity)
            return

        if event.type in _SETTLES:
            kind, field = _SETTLES[event.type]
            identity = self._identity(event, field)
            self._require_started(kind, identity)
            self._require_hierarchy(event, kind, identity)
            if identity in self._settled[kind]:
                raise LifecycleError(f"{kind} {identity} is already settled")
            self._outcome(event, allow_paused=kind == "execution")
            if identity in self._paused[kind]:
                raise LifecycleError(f"paused {kind} {identity} must resume before settlement")
            self._require_closed_children(kind, identity)
            self._settled[kind].add(identity)

    def is_settled(self, kind: _Kind, identity: str) -> bool:
        return identity in self._settled[kind]

    def is_paused(self, kind: _Kind, identity: str) -> bool:
        return identity in self._paused[kind]

    @staticmethod
    def _identity(event: PendingSessionEvent | SessionEvent, field: str) -> str:
        value = getattr(event, field)
        if value is None:
            raise LifecycleError(f"{field} is required for {event.type.value}")
        return value

    @staticmethod
    def _hierarchy(
        event: PendingSessionEvent | SessionEvent, kind: _Kind
    ) -> tuple[str | None, ...]:
        base: tuple[str | None, ...] = (event.session_id, event.task_id)
        if kind == "execution":
            return (*base, event.turn_id)
        if kind in {"submission", "turn"}:
            return base
        if kind == "step":
            return (*base, event.turn_id)
        return (*base, event.turn_id, event.step_id)

    def _require_hierarchy(
        self, event: PendingSessionEvent | SessionEvent, kind: _Kind, identity: str
    ) -> None:
        if self._parents[(kind, identity)] != self._hierarchy(event, kind):
            raise LifecycleError(f"{kind} {identity} hierarchy does not match its start")

    def _require_started(self, kind: _Kind, identity: str) -> None:
        if identity not in self._started[kind]:
            raise LifecycleError(f"{kind} {identity} was not started")

    def _require_closed_children(self, kind: _Kind, identity: str) -> None:
        child_kinds: dict[_Kind, _Kind] = {
            "submission": "turn",
            "turn": "step",
            "step": "attempt",
        }
        child_kind = child_kinds.get(kind)
        if child_kind is None:
            return
        for (candidate_kind, candidate_id), hierarchy in self._parents.items():
            if candidate_kind != child_kind or candidate_id in self._settled[child_kind]:
                continue
            if kind == "submission" and hierarchy[1] == self._parents[(kind, identity)][1]:
                raise LifecycleError(f"{kind} {identity} has an open {child_kind}")
            if kind == "turn" and hierarchy[-1] == identity:
                raise LifecycleError(f"{kind} {identity} has an open {child_kind}")
            if kind == "step" and hierarchy[-1] == identity:
                raise LifecycleError(f"{kind} {identity} has an open {child_kind}")
        if kind == "step":
            for (candidate_kind, candidate_id), hierarchy in self._parents.items():
                if candidate_kind == "tool" and candidate_id not in self._settled["tool"]:
                    if hierarchy[-1] == identity:
                        raise LifecycleError(f"{kind} {identity} has an open tool")

    def _require_open_parent(self, event: PendingSessionEvent | SessionEvent, kind: _Kind) -> None:
        if kind == "step":
            parent_kind: _Kind = "turn"
            parent_id = event.turn_id
        elif kind in {"attempt", "tool"}:
            parent_kind = "step"
            parent_id = event.step_id
        else:
            return
        if parent_id is None or parent_id not in self._started[parent_kind]:
            raise LifecycleError(f"{kind} parent {parent_kind} was not started")
        if parent_id in self._settled[parent_kind]:
            raise LifecycleError(f"{kind} cannot start under a settled {parent_kind}")
        parent_hierarchy = self._parents[(parent_kind, parent_id)]
        child_hierarchy = self._hierarchy(event, kind)
        if child_hierarchy[: len(parent_hierarchy)] != parent_hierarchy:
            raise LifecycleError(f"{kind} parent hierarchy does not match")

    @staticmethod
    def _outcome(event: PendingSessionEvent | SessionEvent, *, allow_paused: bool) -> str:
        outcome = event.payload.get("outcome")
        allowed = _FINAL_OUTCOMES | ({"paused"} if allow_paused else set())
        if not isinstance(outcome, str) or outcome not in allowed:
            raise LifecycleError(f"{event.type.value} requires a valid outcome")
        return outcome


class FactRecorder:
    """Validate and atomically persist facts with their Task snapshot."""

    def __init__(self, store: ExecutionStore, *, limits: ExecutionLimits | None = None) -> None:
        self.store = store
        self.limits = limits or ExecutionLimits()

    def commit(
        self,
        task: Task,
        events: Sequence[PendingSessionEvent],
        submission: SubmissionRecord | None = None,
    ) -> list[SessionEvent]:
        expected = task.last_sequence or 0
        pending = [PendingSessionEvent.model_validate(event.model_dump()) for event in events]
        session_id = self._session_id(task, pending, submission)

        tracker = LifecycleTracker()
        tracker.extend(self.store.events(session_id))
        tracker.extend(pending)

        for offset, event in enumerate(pending, start=1):
            durable_event = SessionEvent(
                sequence=expected + offset,
                **event.model_dump(),
            )
            if _json_size(durable_event.model_dump(mode="json")) > self.limits.max_event_bytes:
                raise PayloadLimitError("event exceeds max_event_bytes")
        snapshot = Task.from_dict(task.to_dict())
        snapshot.last_sequence = expected + len(pending)
        if _json_size(snapshot.to_dict()) > self.limits.max_task_snapshot_bytes:
            raise PayloadLimitError("task snapshot exceeds max_task_snapshot_bytes")

        committed = self.store.commit(
            task,
            pending,
            expected_sequence=expected,
            submission=submission,
        )
        task.last_sequence = committed[-1].sequence if committed else expected
        return committed

    @staticmethod
    def _session_id(
        task: Task,
        events: Sequence[PendingSessionEvent],
        submission: SubmissionRecord | None,
    ) -> str:
        candidates = [event.session_id for event in events]
        if task.session_id is not None:
            candidates.append(task.session_id)
        if submission is not None:
            candidates.append(submission.session_id)
        if not candidates or any(value != candidates[0] for value in candidates):
            raise LifecycleError("fact batch requires one consistent session identity")
        return candidates[0]


def _json_size(value: Any) -> int:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return len(encoded.encode("utf-8"))
