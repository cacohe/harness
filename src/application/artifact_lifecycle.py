import inspect
from dataclasses import replace

from src.core.artifacts.lifecycle import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ArtifactOwnershipStore,
)
from src.core.execution.errors import RunErrorCode
from src.core.ids import RunId, SessionId
from src.core.persistence.journal import (
    JournalCommit,
    JournalError,
    RunJournal,
)
from src.core.persistence.receipt import validate_commit_receipt


class ArtifactLifecycleService:
    """CAS-safe ownership release built only from authoritative journal descriptors."""

    def __init__(
        self,
        *,
        journal: RunJournal,
        ownership: ArtifactOwnershipStore,
        max_conflict_retries: int = 16,
    ) -> None:
        if type(max_conflict_retries) is not int or not 1 <= max_conflict_retries <= 1_024:
            raise ValueError("Artifact lifecycle retry limit is invalid")
        _require_capability(
            ownership,
            (
                "load_artifact_attachments",
                "artifact_has_active_attachments",
                "load_artifact_facts",
            ),
            "Artifact ownership capability is invalid",
        )
        self._journal = journal
        self._ownership = ownership
        self._max_retries = max_conflict_retries

    async def detach(
        self,
        session_id: SessionId,
        run_id: RunId,
        *,
        roles: tuple[ArtifactAttachmentRole, ...] | None = None,
    ) -> tuple[ArtifactAttachmentFact, ...]:
        selected = _selected_roles(roles)
        for _attempt in range(self._max_retries):
            target = await self._journal.find_run_state_by_id(session_id, run_id)
            if target is None or target.session_id != session_id:
                raise ValueError("Run is not owned by this Session")
            if not target.status.terminal:
                raise ValueError("Artifact attachments require a terminal Run")
            active = await self._ownership.load_artifact_attachments(session_id, run_id)
            detached = tuple(
                replace(fact, action=ArtifactAttachmentAction.DETACHED)
                for fact in active
                if selected is None or fact.role in selected
            )
            if not detached:
                return ()
            head = await self._journal.load_head(session_id)
            if head.state is None or head.state.session_id != session_id:
                raise ValueError("Session has no durable head")
            persisted = replace(head.state, journal_version=head.version + 1)
            batch = JournalCommit(persisted, artifact_facts=detached)
            try:
                receipt = await self._journal.commit(batch, head.version)
            except JournalError as error:
                if error.failure.code is RunErrorCode.JOURNAL_CONFLICT:
                    continue
                raise
            validate_commit_receipt(
                receipt,
                batch,
                expected_version=head.version,
                previous_sequence=head.sequence,
            )
            return detached
        raise JournalError(RunErrorCode.JOURNAL_CONFLICT)


def _selected_roles(
    roles: tuple[ArtifactAttachmentRole, ...] | None,
) -> frozenset[ArtifactAttachmentRole] | None:
    if roles is None:
        return None
    if (
        type(roles) is not tuple
        or not roles
        or len(roles) > len(ArtifactAttachmentRole)
        or any(type(role) is not ArtifactAttachmentRole for role in roles)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("Artifact attachment roles are invalid")
    return frozenset(roles)


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
