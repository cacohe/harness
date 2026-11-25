import inspect
from datetime import datetime, timezone

from src.core.artifacts.contracts import ArtifactGarbageCollector, ArtifactRepository
from src.core.artifacts.lifecycle import (
    ArtifactReferenceIndex,
    ArtifactRetirementState,
    ArtifactRetirementStore,
)


class ArtifactRetentionBlocked(RuntimeError):
    """The Artifact still has at least one durable owner."""


class ArtifactRetentionService:
    """Application-level retirement gate; repositories remain storage-only adapters."""

    def __init__(
        self,
        *,
        retirement: ArtifactRetirementStore,
        references: ArtifactReferenceIndex,
        artifacts: ArtifactRepository,
        collector: ArtifactGarbageCollector | None = None,
        max_reference_ids: int = 1_000_000,
    ) -> None:
        if type(max_reference_ids) is not int or not 1 <= max_reference_ids <= 1_000_000:
            raise ValueError("Artifact maintenance limit is invalid")
        _require_capability(
            retirement,
            (
                "reserve_artifact_retirement",
                "complete_artifact_retirement",
                "abort_artifact_retirement",
            ),
            "Artifact retirement capability is invalid",
        )
        _require_capability(
            references,
            ("load_referenced_artifact_ids",),
            "Artifact reference capability is invalid",
        )
        _require_capability(
            artifacts,
            ("stat", "delete"),
            "Artifact repository capability is invalid",
        )
        if collector is not None:
            _require_capability(
                collector,
                ("collect_orphans",),
                "Artifact garbage collector capability is invalid",
            )
        self._retirement = retirement
        self._references = references
        self._artifacts = artifacts
        self._collector = collector
        self._max_reference_ids = max_reference_ids

    async def retire(self, artifact_id: str) -> None:
        if type(artifact_id) is not str or not artifact_id:
            raise ValueError("Artifact id is invalid")
        reservation = await self._retirement.reserve_artifact_retirement(artifact_id)
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
                await self._retirement.complete_artifact_retirement(reservation)
            else:
                await self._retirement.abort_artifact_retirement(reservation)
            raise
        await self._retirement.complete_artifact_retirement(reservation)

    async def collect_orphans(self, older_than: datetime) -> int:
        if (
            type(older_than) is not datetime
            or older_than.tzinfo is not timezone.utc
            or older_than.utcoffset() is None
        ):
            raise ValueError("Artifact orphan cutoff must be an exact UTC datetime")
        if self._collector is None:
            raise RuntimeError("Artifact orphan collection is unavailable")
        references = await self._references.load_referenced_artifact_ids(
            self._max_reference_ids + 1
        )
        if type(references) is not tuple or any(
            type(artifact_id) is not str or not artifact_id for artifact_id in references
        ):
            raise RuntimeError("Artifact reference inventory is invalid")
        if len(references) > self._max_reference_ids:
            raise RuntimeError("Artifact reference inventory exceeds its configured limit")
        result = await self._collector.collect_orphans(references, older_than)
        if type(result) is not int or result < 0:
            raise RuntimeError("Artifact orphan collector returned an invalid result")
        return result


def _require_capability(value: object, names: tuple[str, ...], message: str) -> None:
    for name in names:
        try:
            member = inspect.getattr_static(value, name)
        except (AttributeError, TypeError):
            raise ValueError(message) from None
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        if not callable(member):
            raise ValueError(message)
