from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from src.core.artifacts.contracts import ArtifactRef, snapshot_artifact_ref
from src.core.ids import RunId, SessionId, SubmissionId


class ArtifactAttachmentRole(StrEnum):
    TOOL_RESULT = "tool_result"
    FINAL_OUTPUT = "final_output"
    SUMMARY = "summary"


class ArtifactAttachmentAction(StrEnum):
    ATTACHED = "attached"
    DETACHED = "detached"


class ArtifactRetirementState(StrEnum):
    AVAILABLE = "available"
    RETIRING = "retiring"
    RETIRED = "retired"
    ABORTED = "aborted"


@dataclass(frozen=True)
class ArtifactRetirementReservation:
    artifact_id: str
    generation: int
    state: ArtifactRetirementState

    def __post_init__(self) -> None:
        if (
            type(self.artifact_id) is not str
            or not self.artifact_id
            or type(self.generation) is not int
            or self.generation < 1
            or type(self.state) is not ArtifactRetirementState
        ):
            raise ValueError("Artifact retirement reservation is invalid")


@dataclass(frozen=True)
class ArtifactAttachmentFact:
    """Durable lifecycle fact for one Artifact descriptor; never stores payloads or paths."""

    session_id: SessionId
    run_id: RunId
    submission_id: SubmissionId
    owner_id: str
    role: ArtifactAttachmentRole
    action: ArtifactAttachmentAction
    artifact: ArtifactRef

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not SessionId
            or type(self.run_id) is not RunId
            or type(self.submission_id) is not SubmissionId
            or type(self.owner_id) is not str
            or not self.owner_id
            or self.owner_id.strip() != self.owner_id
            or len(self.owner_id.encode("utf-8")) > 256
            or type(self.role) is not ArtifactAttachmentRole
            or type(self.action) is not ArtifactAttachmentAction
            or type(self.artifact) is not ArtifactRef
        ):
            raise ValueError("Artifact attachment fact is invalid")
        object.__setattr__(self, "session_id", SessionId(self.session_id.value))
        object.__setattr__(self, "run_id", RunId(self.run_id.value))
        object.__setattr__(self, "submission_id", SubmissionId(self.submission_id.value))
        object.__setattr__(self, "artifact", snapshot_artifact_ref(self.artifact))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.owner_id, self.role.value, self.artifact.artifact_id


@dataclass(frozen=True)
class ArtifactFactRecord:
    index: int
    fact: ArtifactAttachmentFact

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or type(self.fact) is not ArtifactAttachmentFact
        ):
            raise ValueError("Artifact fact record is invalid")


class ArtifactOwnershipStore(Protocol):
    def load_artifact_attachments(
        self, session_id: SessionId, run_id: RunId
    ) -> Awaitable[tuple[ArtifactAttachmentFact, ...]]:
        """Return canonical active attachments for one Run using a bounded index lookup."""
        ...

    def artifact_has_active_attachments(self, artifact_id: str) -> Awaitable[bool]:
        """Return whether retirement is forbidden by any active durable attachment."""
        ...

    def load_artifact_facts(
        self,
        session_id: SessionId,
        run_id: RunId,
        after_index: int,
        limit: int,
    ) -> Awaitable[tuple[ArtifactFactRecord, ...]]:
        """Page append-only lifecycle facts for one Run by Session-global fact index."""
        ...


class ArtifactRetirementStore(Protocol):
    def reserve_artifact_retirement(
        self, artifact_id: str
    ) -> Awaitable[ArtifactRetirementReservation | None]:
        """Atomically reserve retirement only when no active attachment exists."""
        ...

    def complete_artifact_retirement(
        self, reservation: ArtifactRetirementReservation
    ) -> Awaitable[ArtifactRetirementReservation]:
        """Complete the exact active retirement generation."""
        ...

    def abort_artifact_retirement(
        self, reservation: ArtifactRetirementReservation
    ) -> Awaitable[ArtifactRetirementReservation]:
        """Release the exact active generation after a confirmed failed delete."""
        ...


class ArtifactReferenceIndex(Protocol):
    def load_referenced_artifact_ids(self, limit: int) -> Awaitable[tuple[str, ...]]:
        """Return a bounded inventory of publication identities still referenced by Runs."""
        ...


class ArtifactLifecycleStore(
    ArtifactOwnershipStore,
    ArtifactRetirementStore,
    ArtifactReferenceIndex,
    Protocol,
):
    """Composite capability implemented by one atomic Artifact lifecycle authority."""
