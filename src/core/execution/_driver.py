from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from src.core.artifacts.contracts import ArtifactRef
from src.core.execution.commands import CommandContext, CoreCommand, FailRun
from src.core.execution.errors import RunErrorCode, RunFailure
from src.core.execution.events import DomainEvent
from src.core.execution.state import RunState, StateTransition, TaskStateMachine
from src.core.persistence.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ConversationSummaryCheckpoint,
    JournalCommit,
    TranscriptEntry,
)
from src.core.persistence.receipt import validate_commit_receipt

if TYPE_CHECKING:
    from src.core.execution.coordinator import RunCoordinator


@dataclass(frozen=True, slots=True)
class _CommittedRun:
    state: RunState
    journal_version: int
    sequence: int
    events: tuple[DomainEvent, ...] = ()


class _Driver:
    def __init__(self, owner: RunCoordinator, state: RunState, supplied: CommandContext) -> None:
        self.owner = owner
        self.state = state
        self.events: list[DomainEvent] = []
        self.transcript: tuple[TranscriptEntry, ...] = ()
        self.checkpoint: ConversationSummaryCheckpoint | None = None
        self.artifact_facts: list[ArtifactAttachmentFact] = []
        self.base_elapsed = supplied.elapsed_seconds
        self.started = owner._clock.monotonic()

    @property
    def run_id(self):
        return self.state.run_id

    @property
    def session_id(self):
        return self.state.session_id

    def context(self) -> CommandContext:
        elapsed = self.elapsed_seconds()
        return CommandContext(self.owner._clock.now(), elapsed, (self.owner._ids.event(),))

    def elapsed_seconds(self) -> float:
        return self.base_elapsed + max(0.0, self.owner._clock.monotonic() - self.started)

    def remaining_seconds(self) -> float:
        return max(0.0, self.state.limits.max_elapsed_seconds - self.elapsed_seconds())

    def apply(self, command: CoreCommand) -> bool:
        transition: StateTransition = TaskStateMachine.apply(self.state, command)
        self.state = transition.state
        self.events.extend(transition.events)
        return self.state.status.terminal

    async def fail(self, code: RunErrorCode) -> _CommittedRun:
        self.checkpoint = None
        self.apply(FailRun(self.run_id, self.session_id, RunFailure(code), context=self.context()))
        return await self.commit()

    def entry(self, role: str, content: str) -> TranscriptEntry:
        return TranscriptEntry(
            self.state.session_id,
            self.state.run_id,
            self.state.submission_id,
            role,
            content,
        )

    def attach_refs(
        self, refs: tuple[ArtifactRef, ...], role: ArtifactAttachmentRole, owner_id: str
    ) -> None:
        self.artifact_facts.extend(
            ArtifactAttachmentFact(
                self.state.session_id,
                self.state.run_id,
                self.state.submission_id,
                owner_id,
                role,
                ArtifactAttachmentAction.ATTACHED,
                ref,
            )
            for ref in refs
        )

    async def commit(self) -> _CommittedRun:
        expected = self.state.journal_version
        sequence = self.state.journal_sequence + len(self.events) + len(self.transcript)
        persisted = replace(self.state, journal_version=expected + 1, journal_sequence=sequence)
        facts = list(self.artifact_facts)
        if self.checkpoint is not None:
            owner = (
                f"{self.checkpoint.policy_version}:"
                f"{self.checkpoint.covered_from_sequence}-{self.checkpoint.covered_through_sequence}"
            )
            known = {(fact.role, fact.owner_id, fact.artifact.artifact_id) for fact in facts}
            for ref in self.checkpoint.artifacts:
                key = (ArtifactAttachmentRole.SUMMARY, owner, ref.artifact_id)
                if key not in known:
                    facts.append(
                        ArtifactAttachmentFact(
                            persisted.session_id,
                            persisted.run_id,
                            persisted.submission_id,
                            owner,
                            ArtifactAttachmentRole.SUMMARY,
                            ArtifactAttachmentAction.ATTACHED,
                            ref,
                        )
                    )
        batch = JournalCommit(
            persisted,
            events=tuple(self.events),
            transcript=self.transcript,
            summary=self.checkpoint,
            artifact_facts=tuple(facts),
        )
        receipt = await self.owner._journal.commit(batch, expected)
        receipt = validate_commit_receipt(
            receipt,
            batch,
            expected_version=expected,
            previous_sequence=self.state.journal_sequence,
        )
        return _CommittedRun(persisted, receipt.version, receipt.sequence, tuple(self.events))
