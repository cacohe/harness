import asyncio
from dataclasses import replace

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.artifact_retention import ArtifactRetentionBlocked, ArtifactRetentionService
from src.core.artifacts import ArtifactWrite
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    JournalCommit,
)
from tests.unit.core.test_journal_contract import acceptance


def test_retirement_requires_all_durable_attachments_to_be_detached() -> None:
    async def scenario() -> None:
        async def payload():
            yield b"payload"

        journal = InMemoryRunJournal()
        repository = InMemoryArtifactRepository(1024)
        ref = await repository.put(ArtifactWrite(payload(), "text/plain"))
        batch = acceptance("retention")
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

        state = replace(batch.state, version=2, journal_version=2)
        await journal.commit(
            JournalCommit(
                state,
                artifact_facts=(replace(fact, action=ArtifactAttachmentAction.DETACHED),),
            ),
            1,
        )
        with pytest.raises(ArtifactRetentionBlocked):
            await service.retire(ref.artifact_id)
        other_state = replace(other.state, version=2, journal_version=2)
        await journal.commit(
            JournalCommit(
                other_state,
                artifact_facts=(
                    replace(other_fact, action=ArtifactAttachmentAction.DETACHED),
                ),
            ),
            1,
        )
        await service.retire(ref.artifact_id)
        assert await repository.stat(ref.artifact_id) is None

    asyncio.run(scenario())
