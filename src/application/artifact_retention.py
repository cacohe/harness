from src.core.artifacts import ArtifactRepository
from src.core.journal import RunJournal


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
        if await self._journal.artifact_has_active_attachments(artifact_id):
            raise ArtifactRetentionBlocked("Artifact has active attachments")
        await self._artifacts.delete(artifact_id)
