import inspect
from collections.abc import Awaitable
from typing import Protocol, get_type_hints

import pytest
from src.core.artifacts.contracts import ArtifactGarbageCollector, ArtifactRef
from src.core.artifacts.lifecycle import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ArtifactFactRecord,
    ArtifactLifecycleStore,
    ArtifactOwnershipStore,
    ArtifactReferenceIndex,
    ArtifactRetirementReservation,
    ArtifactRetirementState,
    ArtifactRetirementStore,
)
from src.core.ids import RunId, SessionId, SubmissionId
from src.core.persistence.journal import RunJournal


def _artifact_ref() -> ArtifactRef:
    digest = "a" * 64
    return ArtifactRef(f"sha256:{digest}", "text/plain", 1, digest)


def test_artifact_lifecycle_ports_own_the_complete_read_contract() -> None:
    assert set(ArtifactOwnershipStore.__dict__) >= {
        "load_artifact_attachments",
        "artifact_has_active_attachments",
        "load_artifact_facts",
    }
    assert set(ArtifactRetirementStore.__dict__) >= {
        "reserve_artifact_retirement",
        "complete_artifact_retirement",
        "abort_artifact_retirement",
    }
    assert set(ArtifactReferenceIndex.__dict__) >= {"load_referenced_artifact_ids"}
    assert ArtifactLifecycleStore.__bases__ == (
        ArtifactOwnershipStore,
        ArtifactRetirementStore,
        ArtifactReferenceIndex,
        Protocol,
    )


def test_artifact_lifecycle_port_signatures_are_bounded() -> None:
    assert tuple(inspect.signature(ArtifactOwnershipStore.load_artifact_facts).parameters) == (
        "self",
        "session_id",
        "run_id",
        "after_index",
        "limit",
    )
    hints = get_type_hints(ArtifactReferenceIndex.load_referenced_artifact_ids)
    assert hints == {"limit": int, "return": Awaitable[tuple[str, ...]]}
    assert tuple(inspect.signature(ArtifactGarbageCollector.collect_orphans).parameters) == (
        "self",
        "referenced_ids",
        "older_than",
    )


def test_run_journal_declares_only_atomic_artifact_mutation() -> None:
    for member in (
        "load_artifact_attachments",
        "artifact_has_active_attachments",
        "reserve_artifact_retirement",
        "complete_artifact_retirement",
        "abort_artifact_retirement",
        "load_artifact_facts",
        "load_referenced_artifact_ids",
    ):
        assert member not in RunJournal.__dict__


def test_artifact_lifecycle_values_preserve_validation() -> None:
    fact = ArtifactAttachmentFact(
        SessionId("session"),
        RunId("run"),
        SubmissionId("submission"),
        "owner",
        ArtifactAttachmentRole.TOOL_RESULT,
        ArtifactAttachmentAction.ATTACHED,
        _artifact_ref(),
    )
    assert fact.key == ("owner", "tool_result", fact.artifact.artifact_id)
    assert ArtifactFactRecord(0, fact).fact == fact
    reservation = ArtifactRetirementReservation(
        fact.artifact.artifact_id,
        1,
        ArtifactRetirementState.RETIRING,
    )
    assert reservation.generation == 1

    with pytest.raises(ValueError, match="reservation is invalid"):
        ArtifactRetirementReservation(fact.artifact.artifact_id, 0, reservation.state)
    with pytest.raises(ValueError, match="attachment fact is invalid"):
        ArtifactAttachmentFact(
            fact.session_id,
            fact.run_id,
            fact.submission_id,
            " owner ",
            fact.role,
            fact.action,
            fact.artifact,
        )
    with pytest.raises(ValueError, match="fact record is invalid"):
        ArtifactFactRecord(-1, fact)
