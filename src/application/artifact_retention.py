from src.core.artifacts import ArtifactRepository
from src.core.journal import ArtifactRetirementState, RunJournal


class ArtifactRetentionBlocked(RuntimeError):
    """The Artifact still has at least one durable owner."""


class ArtifactRetentionService:
    """Application-level retirement gate; repositories remain storage-only adapters."""

    def __init__(self, *, journal: RunJournal, artifacts: ArtifactRepository) -> None:
        self._journal = journal
        self._artifacts = artifacts

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
