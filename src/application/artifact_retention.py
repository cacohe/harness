from datetime import datetime, timezone

from src.core.artifacts import ArtifactRepository
from src.core.journal import ArtifactRetirementState, RunJournal


class ArtifactRetentionBlocked(RuntimeError):
    """The Artifact still has at least one durable owner."""


class ArtifactRetentionService:
    """Application-level retirement gate; repositories remain storage-only adapters."""

    def __init__(
        self,
        *,
        journal: RunJournal,
        artifacts: ArtifactRepository,
        max_reference_ids: int = 1_000_000,
    ) -> None:
        if type(max_reference_ids) is not int or not 1 <= max_reference_ids <= 1_000_000:
            raise ValueError("Artifact maintenance limit is invalid")
        self._journal = journal
        self._artifacts = artifacts
        self._max_reference_ids = max_reference_ids

    async def retire(self, artifact_id: str) -> None:
        if type(artifact_id) is not str or not artifact_id:
            raise ValueError("Artifact id is invalid")
        reservation = await self._journal.reserve_artifact_retirement(artifact_id)
        if reservation is None:
            raise ArtifactRetentionBlocked("Artifact has active attachments")
        if reservation.state is ArtifactRetirementState.RETIRED:
            return
        try:
            await self._artifacts.delete(artifact_id)
        except Exception:
            try:
                current = await self._artifacts.stat(artifact_id)
            except Exception:
                raise
            if current is None:
                await self._journal.complete_artifact_retirement(reservation)
            else:
                await self._journal.abort_artifact_retirement(reservation)
            raise
        await self._journal.complete_artifact_retirement(reservation)

    async def collect_orphans(self, older_than: datetime) -> int:
        if (
            type(older_than) is not datetime
            or older_than.tzinfo is not timezone.utc
            or older_than.utcoffset() is None
        ):
            raise ValueError("Artifact orphan cutoff must be an exact UTC datetime")
        load_references = getattr(self._journal, "load_referenced_artifact_ids", None)
        collect = getattr(self._artifacts, "collect_orphans", None)
        if not callable(load_references) or not callable(collect):
            raise RuntimeError("Artifact orphan collection is unavailable")
        references = await load_references(self._max_reference_ids + 1)
        if (
            type(references) is not tuple
            or any(type(artifact_id) is not str or not artifact_id for artifact_id in references)
        ):
            raise RuntimeError("Artifact reference inventory is invalid")
        if len(references) > self._max_reference_ids:
            raise RuntimeError("Artifact reference inventory exceeds its configured limit")
        result = await collect(references, older_than)
        if type(result) is not int or result < 0:
            raise RuntimeError("Artifact orphan collector returned an invalid result")
        return result
