from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.artifacts.contracts import ArtifactRef, ArtifactRepository, snapshot_artifact_ref
from src.core.execution.commands import CommandContext, FailRun
from src.core.execution.errors import RunErrorCode, RunFailure
from src.core.execution.state import RunState, RunStatus, TaskStateMachine
from src.core.ids import Clock, IdGenerator, SessionId
from src.core.persistence.journal import JournalCommit, RunJournal
from src.core.persistence.receipt import validate_commit_receipt


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    state: RunState | None
    changed: bool


class RecoveryService:
    """Close interrupted durable work without replaying any external operation."""

    def __init__(
        self,
        *,
        journal: RunJournal,
        artifacts: ArtifactRepository,
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._journal = journal
        self._artifacts = artifacts
        self._clock = clock
        self._ids = ids

    async def recover_session(self, session_id: SessionId) -> RecoveryOutcome:
        head = await self._journal.load_head(session_id)
        state = head.state
        if state is None:
            return RecoveryOutcome(None, False)
        attachments = await self._journal.load_artifact_attachments(session_id, state.run_id)
        refs = tuple(fact.artifact for fact in attachments)
        missing = await self._has_dangling_ref(refs)
        if state.status.terminal:
            if missing:
                if (
                    state.status is RunStatus.FAILED
                    and state.failure is not None
                    and state.failure.code is RunErrorCode.ARTIFACT_FAILED
                ):
                    return RecoveryOutcome(state, False)
                raise RecoveryCorruption(RunErrorCode.ARTIFACT_FAILED)
            return RecoveryOutcome(state, False)
        if state.status is RunStatus.WAITING_APPROVAL and not missing:
            return RecoveryOutcome(state, False)

        code = RunErrorCode.ARTIFACT_FAILED if missing else _interrupted_code(state)
        context = CommandContext(self._clock.now(), state.elapsed_seconds, (self._ids.event(),))
        transition = TaskStateMachine.apply(
            state,
            FailRun(state.run_id, state.session_id, RunFailure(code), context=context),
        )
        next_sequence = head.sequence + len(transition.events)
        persisted = replace(
            transition.state,
            journal_version=head.version + 1,
            journal_sequence=next_sequence,
        )
        batch = JournalCommit(persisted, events=transition.events)
        receipt = await self._journal.commit(batch, head.version)
        validate_commit_receipt(
            receipt,
            batch,
            expected_version=head.version,
            previous_sequence=head.sequence,
        )
        return RecoveryOutcome(persisted, True)

    async def _has_dangling_ref(self, refs: tuple[ArtifactRef, ...]) -> bool:
        expected_by_id: dict[str, ArtifactRef] = {}
        for ref in refs:
            try:
                expected = snapshot_artifact_ref(ref)
                prior = expected_by_id.get(expected.artifact_id)
                if prior is not None and prior != expected:
                    return True
                expected_by_id[expected.artifact_id] = expected
            except Exception:
                return True
        for expected in expected_by_id.values():
            try:
                current = await self._artifacts.stat(expected.artifact_id)
                if current is None or type(current) is not ArtifactRef:
                    return True
                actual = snapshot_artifact_ref(current)
            except Exception:
                return True
            if actual != expected:
                return True
        return False


class RecoveryCorruption(RuntimeError):
    def __init__(self, code: RunErrorCode) -> None:
        self.failure = RunFailure(code)
        super().__init__(self.failure.message)


def _interrupted_code(state: RunState) -> RunErrorCode:
    step = state.current_step
    if step is not None and step.model_call is not None and step.model_call.tool_requests:
        return RunErrorCode.TOOL_FAILED
    return RunErrorCode.MODEL_FAILED
