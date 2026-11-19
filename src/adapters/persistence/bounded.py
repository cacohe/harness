from __future__ import annotations

from src.adapters.persistence.sqlite_codec import encode
from src.core.context import ConversationPage
from src.core.errors import RunErrorCode
from src.core.ids import ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ArtifactAttachmentFact,
    ArtifactFactRecord,
    ArtifactRetirementReservation,
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    JournalHead,
    JournalSnapshot,
    RunJournal,
    SubmissionAcceptance,
    TranscriptEntry,
)
from src.core.state import RunState


class BoundedRunJournal:
    """Reject a RunState whose canonical durable projection exceeds its envelope."""

    def __init__(self, delegate: RunJournal, max_state_bytes: int) -> None:
        if type(max_state_bytes) is not int or not 1_024 <= max_state_bytes <= 16 * 1_048_576:
            raise ValueError("Serialized RunState envelope is invalid")
        self._delegate = delegate
        self.max_state_bytes = max_state_bytes

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        if type(batch) is not JournalCommit:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        try:
            size = len(encode(batch.state).encode("utf-8"))
        except (AttributeError, MemoryError, RecursionError, TypeError, ValueError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        if size > self.max_state_bytes:
            raise JournalError(RunErrorCode.LIMIT_EXCEEDED)
        return await self._delegate.commit(batch, expected_version)

    async def load_head(self, session_id: SessionId) -> JournalHead:
        return await self._delegate.load_head(session_id)

    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        return await self._delegate.load_session(session_id, after_sequence)

    async def load_context_summaries(
        self,
        session_id: SessionId,
        profile_id: ProfileId,
        policy_version: str,
        covered_through_sequence: int,
        limit: int,
        expected_version: int,
        expected_sequence: int,
    ) -> tuple[ConversationSummaryCheckpoint, ...]:
        return await self._delegate.load_context_summaries(
            session_id,
            profile_id,
            policy_version,
            covered_through_sequence,
            limit,
            expected_version,
            expected_sequence,
        )

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionAcceptance | None:
        return await self._delegate.find_submission(session_id, submission_id)

    async def find_run_input(
        self,
        session_id: SessionId,
        run_id: RunId,
        submission_id: SubmissionId,
    ) -> TranscriptEntry | None:
        return await self._delegate.find_run_input(session_id, run_id, submission_id)

    async def read_page(
        self, session_id: SessionId, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        return await self._delegate.read_page(session_id, before_sequence, limit)

    async def find_run_state(
        self,
        session_id: SessionId,
        run_id: RunId,
        submission_id: SubmissionId,
    ) -> RunState | None:
        return await self._delegate.find_run_state(session_id, run_id, submission_id)

    async def find_run_state_by_id(self, session_id: SessionId, run_id: RunId) -> RunState | None:
        return await self._delegate.find_run_state_by_id(session_id, run_id)

    async def load_artifact_attachments(
        self, session_id: SessionId, run_id: RunId
    ) -> tuple[ArtifactAttachmentFact, ...]:
        return await self._delegate.load_artifact_attachments(session_id, run_id)

    async def artifact_has_active_attachments(self, artifact_id: str) -> bool:
        return await self._delegate.artifact_has_active_attachments(artifact_id)

    async def reserve_artifact_retirement(
        self, artifact_id: str
    ) -> ArtifactRetirementReservation | None:
        return await self._delegate.reserve_artifact_retirement(artifact_id)

    async def complete_artifact_retirement(
        self, reservation: ArtifactRetirementReservation
    ) -> ArtifactRetirementReservation:
        return await self._delegate.complete_artifact_retirement(reservation)

    async def abort_artifact_retirement(
        self, reservation: ArtifactRetirementReservation
    ) -> ArtifactRetirementReservation:
        return await self._delegate.abort_artifact_retirement(reservation)

    async def load_artifact_facts(
        self,
        session_id: SessionId,
        run_id: RunId,
        after_index: int,
        limit: int,
    ) -> tuple[ArtifactFactRecord, ...]:
        return await self._delegate.load_artifact_facts(session_id, run_id, after_index, limit)
