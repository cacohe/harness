from __future__ import annotations

from dataclasses import dataclass

from src.core.context import (
    ContextRequest,
    ConversationCompactor,
    ConversationInteraction,
    SummaryCompactionRequest,
    snapshot_summary_checkpoint,
)
from src.core.journal import ConversationSummaryCheckpoint


@dataclass(frozen=True, slots=True)
class CompactionOutcome:
    """A non-durable candidate plus the original records, which remain authoritative."""

    checkpoint: ConversationSummaryCheckpoint | None
    retained: tuple[ConversationInteraction, ...]


class SummaryCheckpointService:
    def __init__(self, compactor: ConversationCompactor | None = None) -> None:
        self._compactor = compactor

    def find_latest_reusable(
        self,
        checkpoints: tuple[ConversationSummaryCheckpoint, ...],
        request: ContextRequest,
    ) -> ConversationSummaryCheckpoint | None:
        reusable: list[ConversationSummaryCheckpoint] = []
        for candidate in checkpoints:
            try:
                checkpoint = snapshot_summary_checkpoint(
                    candidate,
                    session_id=request.session_id,
                    profile_id=request.profile_id,
                    policy_version=request.policy_version,
                )
            except ValueError:
                continue
            assert checkpoint is not None
            reusable.append(checkpoint)
        return max(reusable, key=lambda item: item.covered_through_sequence, default=None)

    def find_reusable(
        self,
        checkpoints: tuple[ConversationSummaryCheckpoint, ...],
        request: ContextRequest,
        covered_through_sequence: int,
    ) -> ConversationSummaryCheckpoint | None:
        if type(checkpoints) is not tuple or type(covered_through_sequence) is not int:
            raise ValueError("Checkpoint lookup is invalid")
        for candidate in checkpoints:
            try:
                checkpoint = snapshot_summary_checkpoint(
                    candidate,
                    session_id=request.session_id,
                    profile_id=request.profile_id,
                    policy_version=request.policy_version,
                )
            except ValueError:
                continue
            assert checkpoint is not None
            if checkpoint.covered_through_sequence == covered_through_sequence:
                return checkpoint
        return None

    async def compact_or_retain(
        self,
        request: ContextRequest,
        interactions: tuple[ConversationInteraction, ...],
    ) -> CompactionOutcome:
        compaction = SummaryCompactionRequest(
            request.session_id,
            request.profile_id,
            request.policy_version,
            interactions,
        )
        retained = compaction.interactions
        if self._compactor is None:
            return CompactionOutcome(None, retained)
        try:
            candidate = await self._compactor.compact(compaction)
            checkpoint = snapshot_summary_checkpoint(
                candidate,
                session_id=request.session_id,
                profile_id=request.profile_id,
                policy_version=request.policy_version,
            )
            assert checkpoint is not None
            if (
                checkpoint.covered_from_sequence != retained[0].sequence_from
                or checkpoint.covered_through_sequence != retained[-1].sequence_through
            ):
                raise ValueError
        except Exception:
            return CompactionOutcome(None, retained)
        return CompactionOutcome(checkpoint, retained)
