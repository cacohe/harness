import asyncio
from dataclasses import replace
from typing import cast

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.artifact_lifecycle import ArtifactLifecycleService
from src.application.artifact_retention import ArtifactRetentionBlocked, ArtifactRetentionService
from src.core.artifacts.contracts import ArtifactRef, ArtifactRepository, ArtifactWrite
from src.core.artifacts.lifecycle import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
)
from src.core.execution.errors import RunErrorCode
from src.core.execution.state import RunStatus
from src.core.persistence.journal import JournalError
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


def test_retiring_and_retired_tombstones_block_attachment_resurrection() -> None:
    async def scenario() -> None:
        class BlockingRepository:
            def __init__(self, ref: ArtifactRef) -> None:
                self.ref = ref
                self.present = True
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def delete(self, artifact_id: str) -> None:
                assert artifact_id == self.ref.artifact_id
                self.started.set()
                await self.release.wait()
                self.present = False

            async def stat(self, artifact_id: str):
                return self.ref if self.present and artifact_id == self.ref.artifact_id else None

        journal = InMemoryRunJournal()
        ref = ArtifactRef("sha256:" + "7" * 64, "text/plain", 1, "7" * 64)
        repository = BlockingRepository(ref)
        retirement = asyncio.create_task(
            ArtifactRetentionService(
                journal=journal, artifacts=cast(ArtifactRepository, repository)
            ).retire(ref.artifact_id)
        )
        await repository.started.wait()
        batch = acceptance("attach-during-retire")
        fact = ArtifactAttachmentFact(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "owner",
            ArtifactAttachmentRole.TOOL_RESULT,
            ArtifactAttachmentAction.ATTACHED,
            ref,
        )
        with pytest.raises(JournalError) as caught:
            await journal.commit(replace(batch, artifact_facts=(fact,)), 0)
        assert caught.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
        repository.release.set()
        await retirement
        repository.present = True  # content-addressed payload was put again
        with pytest.raises(JournalError) as retired:
            await journal.commit(replace(batch, artifact_facts=(fact,)), 0)
        assert retired.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
        assert not await journal.artifact_has_active_attachments(ref.artifact_id)

    asyncio.run(scenario())


def test_delete_failure_aborts_reservation_and_allows_attach() -> None:
    async def scenario() -> None:
        ref = ArtifactRef("sha256:" + "8" * 64, "text/plain", 1, "8" * 64)

        class FailingRepository:
            async def delete(self, artifact_id: str) -> None:
                raise RuntimeError("delete failed")

            async def stat(self, artifact_id: str):
                return ref

        journal = InMemoryRunJournal()
        service = ArtifactRetentionService(
            journal=journal, artifacts=cast(ArtifactRepository, FailingRepository())
        )
        with pytest.raises(RuntimeError, match="delete failed"):
            await service.retire(ref.artifact_id)
        batch = acceptance("attach-after-abort")
        fact = ArtifactAttachmentFact(
            batch.state.session_id,
            batch.state.run_id,
            batch.state.submission_id,
            "owner",
            ArtifactAttachmentRole.SUMMARY,
            ArtifactAttachmentAction.ATTACHED,
            ref,
        )
        await journal.commit(replace(batch, artifact_facts=(fact,)), 0)

    asyncio.run(scenario())


@pytest.mark.parametrize("crash_point", ("before-delete", "before-complete"))
def test_sqlite_restart_resumes_retirement_saga(tmp_path, crash_point: str) -> None:
    async def scenario() -> None:
        async def payload():
            yield b"restart"

        path = (tmp_path / f"retire-{crash_point}.sqlite3").resolve()
        journal = SQLiteRunJournal(path)
        repository = InMemoryArtifactRepository(1024)
        ref = await repository.put(ArtifactWrite(payload(), "text/plain"))
        reservation = await journal.reserve_artifact_retirement(ref.artifact_id)
        assert reservation is not None
        if crash_point == "before-complete":
            await repository.delete(ref.artifact_id)
        reopened = SQLiteRunJournal(path)
        service = ArtifactRetentionService(journal=reopened, artifacts=repository)
        await service.retire(ref.artifact_id)
        await service.retire(ref.artifact_id)
        assert await repository.stat(ref.artifact_id) is None

    asyncio.run(scenario())
