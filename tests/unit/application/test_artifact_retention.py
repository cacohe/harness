import asyncio
from dataclasses import replace

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.artifact_lifecycle import ArtifactLifecycleService
from src.application.artifact_retention import ArtifactRetentionBlocked, ArtifactRetentionService
from src.core.artifacts import ArtifactWrite
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
)
from src.core.state import RunStatus
from tests.unit.core.test_journal_contract import acceptance


def test_retirement_requires_all_durable_attachments_to_be_detached() -> None:
    async def scenario() -> None:
        async def payload():
            yield b"payload"

        journal = InMemoryRunJournal()
        repository = InMemoryArtifactRepository(1024)
        ref = await repository.put(ArtifactWrite(payload(), "text/plain"))
        batch = acceptance("retention")
        batch = replace(
            batch,
            state=replace(batch.state, status=RunStatus.COMPLETED, final_text="done"),
        )
        fact = ArtifactAttachmentFact(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "owner",
            ArtifactAttachmentRole.TOOL_RESULT,
            ArtifactAttachmentAction.ATTACHED,
            ref,
        )
        await journal.commit(replace(batch, artifact_facts=(fact,)), 0)
        other = acceptance("retention-other", session="other")
        other = replace(
            other,
            state=replace(other.state, status=RunStatus.COMPLETED, final_text="done"),
        )
        other_fact = ArtifactAttachmentFact(
            other.state.session_id,
            other.state.run_id,
            other.state.submission_id,
            "summary-owner",
            ArtifactAttachmentRole.SUMMARY,
            ArtifactAttachmentAction.ATTACHED,
            ref,
        )
        await journal.commit(replace(other, artifact_facts=(other_fact,)), 0)
        service = ArtifactRetentionService(journal=journal, artifacts=repository)
        with pytest.raises(ArtifactRetentionBlocked):
            await service.retire(ref.artifact_id)
        assert await repository.stat(ref.artifact_id) == ref

        lifecycle = ArtifactLifecycleService(journal=journal)
        await lifecycle.detach(batch.state.session_id, batch.state.run_id)
        with pytest.raises(ArtifactRetentionBlocked):
            await service.retire(ref.artifact_id)
        await lifecycle.detach(other.state.session_id, other.state.run_id)
        await service.retire(ref.artifact_id)
        assert await repository.stat(ref.artifact_id) is None

    asyncio.run(scenario())


def test_concurrent_detach_and_retire_never_deletes_an_active_owner() -> None:
    async def scenario() -> None:
        async def payload():
            yield b"concurrent"

        journal = InMemoryRunJournal()
        repository = InMemoryArtifactRepository(1024)
        ref = await repository.put(ArtifactWrite(payload(), "text/plain"))
        batch = acceptance("concurrent-retention")
        batch = replace(
            batch,
            state=replace(batch.state, status=RunStatus.COMPLETED, final_text="done"),
        )
        fact = ArtifactAttachmentFact(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "owner",
            ArtifactAttachmentRole.FINAL_OUTPUT,
            ArtifactAttachmentAction.ATTACHED,
            ref,
        )
        await journal.commit(replace(batch, artifact_facts=(fact,)), 0)
        lifecycle = ArtifactLifecycleService(journal=journal)
        retention = ArtifactRetentionService(journal=journal, artifacts=repository)
        detached, retired = await asyncio.gather(
            lifecycle.detach(batch.state.session_id, batch.state.run_id),
            retention.retire(ref.artifact_id),
            return_exceptions=True,
        )
        assert not isinstance(detached, BaseException)
        if isinstance(retired, ArtifactRetentionBlocked):
            await retention.retire(ref.artifact_id)
        else:
            assert retired is None
        assert await repository.stat(ref.artifact_id) is None
        assert not await journal.artifact_has_active_attachments(ref.artifact_id)

    asyncio.run(scenario())
