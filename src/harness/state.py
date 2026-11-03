from __future__ import annotations

from dataclasses import replace

from harness.types import TERMINAL_STATUSES, RunSnapshot, RunStatus


class IllegalTransition(RuntimeError):
    def __init__(self, src: RunStatus, dst: RunStatus) -> None:
        super().__init__(f"illegal transition {src.value} -> {dst.value}")
        self.src = src
        self.dst = dst


_FLOW: dict[RunStatus, set[RunStatus]] = {
    RunStatus.IDLE: {RunStatus.QUEUED},
    RunStatus.QUEUED: {RunStatus.VALIDATING},
    RunStatus.VALIDATING: {RunStatus.BUILDING_CONTEXT},
    RunStatus.BUILDING_CONTEXT: {RunStatus.INFERRING},
    RunStatus.INFERRING: {RunStatus.DECIDING},
    RunStatus.DECIDING: {
        RunStatus.AWAITING_APPROVAL,
        RunStatus.CHECKING_SAFETY,
        RunStatus.EXECUTING_TOOLS,
        RunStatus.EMITTING,
        RunStatus.QUEUED,
    },
    RunStatus.AWAITING_APPROVAL: {RunStatus.CHECKING_SAFETY, RunStatus.QUEUED},
    RunStatus.CHECKING_SAFETY: {
        RunStatus.EXECUTING_TOOLS,
        RunStatus.QUEUED,
        RunStatus.AWAITING_APPROVAL,
    },
    RunStatus.EXECUTING_TOOLS: {RunStatus.QUEUED},
    RunStatus.EMITTING: {RunStatus.FINISHED},
    RunStatus.FINISHED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}

ALLOWED: dict[RunStatus, set[RunStatus]] = {}
for _status, _targets in _FLOW.items():
    if _status in TERMINAL_STATUSES:
        ALLOWED[_status] = set()
    else:
        ALLOWED[_status] = set(_targets) | {RunStatus.FAILED, RunStatus.CANCELLED}


def transition(snapshot: RunSnapshot, new: RunStatus) -> RunSnapshot:
    if snapshot.status is new:
        return snapshot
    if new not in ALLOWED[snapshot.status]:
        raise IllegalTransition(snapshot.status, new)
    return replace(snapshot, status=new)


class InMemoryStateStore:
    def __init__(self) -> None:
        self._current: dict[str, RunSnapshot] = {}
        self.history: list[RunSnapshot] = []

    def save(self, snapshot: RunSnapshot) -> None:
        stored = replace(
            snapshot,
            messages=list(snapshot.messages),
            last_tool_results=list(snapshot.last_tool_results),
        )
        self._current[snapshot.run_id] = stored
        self.history.append(stored)

    def load(self, run_id: str) -> RunSnapshot | None:
        return self._current.get(run_id)
