import asyncio
from dataclasses import replace

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.artifact_lifecycle import ArtifactLifecycleService
from src.core.artifacts.lifecycle import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
)
from src.core.execution.errors import RunErrorCode
from src.core.execution.state import RunStatus
from src.core.persistence.journal import (
    JournalCommit,
    JournalError,
)
from tests.unit.core.test_journal_contract import acceptance, artifact_ref


def test_detach_uses_authoritative_descriptors_filters_roles_and_is_idempotent() -> None:
    async def scenario() -> None:
        journal = InMemoryRunJournal()
        batch = acceptance("lifecycle")
        terminal = replace(batch.state, status=RunStatus.COMPLETED, final_text="done")
        tool = ArtifactAttachmentFact(
            terminal.session_id,
            terminal.run_id,
            terminal.submission_id,
            "tool",
            ArtifactAttachmentRole.TOOL_RESULT,
            ArtifactAttachmentAction.ATTACHED,
            artifact_ref("b"),
        )
        final = replace(
            tool,
            owner_id="final",
            role=ArtifactAttachmentRole.FINAL_OUTPUT,
            artifact=artifact_ref("c"),
        )
        summary = replace(
            tool,
            owner_id="summary",
            role=ArtifactAttachmentRole.SUMMARY,
            artifact=artifact_ref("d"),
        )
        await journal.commit(
            replace(batch, state=terminal, artifact_facts=(tool, final, summary)), 0
        )
        lifecycle = ArtifactLifecycleService(journal=journal)

        detached = await lifecycle.detach(
            terminal.session_id,
            terminal.run_id,
            roles=(ArtifactAttachmentRole.TOOL_RESULT,),
        )
        assert detached == (replace(tool, action=ArtifactAttachmentAction.DETACHED),)
        assert await journal.load_artifact_attachments(terminal.session_id, terminal.run_id) == (
            final,
            summary,
        )
        assert (
            await lifecycle.detach(
                terminal.session_id,
                terminal.run_id,
                roles=(ArtifactAttachmentRole.TOOL_RESULT,),
            )
            == ()
        )

        remaining = await lifecycle.detach(terminal.session_id, terminal.run_id)
        assert remaining == (
            replace(final, action=ArtifactAttachmentAction.DETACHED),
            replace(summary, action=ArtifactAttachmentAction.DETACHED),
        )

    asyncio.run(scenario())


def test_detach_rejects_non_terminal_or_wrong_run() -> None:
    async def scenario() -> None:
        journal = InMemoryRunJournal()
        batch = acceptance("open-lifecycle")
        await journal.commit(batch, 0)
        lifecycle = ArtifactLifecycleService(journal=journal)
        with pytest.raises(ValueError):
            await lifecycle.detach(batch.state.session_id, batch.state.run_id)
        with pytest.raises(ValueError):
            await lifecycle.detach(batch.state.session_id, type(batch.state.run_id)("missing"))

    asyncio.run(scenario())


def test_sqlite_restart_detaches_historical_owner_without_releasing_shared_digest(
    tmp_path,
) -> None:
    async def scenario() -> None:
        path = (tmp_path / "lifecycle.sqlite3").resolve()
        journal = SQLiteRunJournal(path)
        shared = artifact_ref("e")
        first = acceptance("shared-first")
        first_state = replace(first.state, status=RunStatus.COMPLETED, final_text="first")
        first_fact = ArtifactAttachmentFact(
            first_state.session_id,
            first_state.run_id,
            first_state.submission_id,
            "first-owner",
            ArtifactAttachmentRole.FINAL_OUTPUT,
            ArtifactAttachmentAction.ATTACHED,
            shared,
        )
        await journal.commit(replace(first, state=first_state, artifact_facts=(first_fact,)), 0)
        second = acceptance("shared-second")
        second_state = replace(
            second.state,
            status=RunStatus.COMPLETED,
            final_text="second",
            journal_version=2,
            journal_sequence=4,
        )
        second_fact = replace(
            first_fact,
            run_id=second_state.run_id,
            submission_id=second_state.submission_id,
            owner_id="second-owner",
        )
        await journal.commit(
            replace(
                second,
                state=second_state,
                summary=None,
                artifact_facts=(second_fact,),
            ),
            1,
        )

        reopened = SQLiteRunJournal(path)
        lifecycle = ArtifactLifecycleService(journal=reopened)
        detached = await lifecycle.detach(first_state.session_id, first_state.run_id)
        assert detached == (replace(first_fact, action=ArtifactAttachmentAction.DETACHED),)
        assert await reopened.load_artifact_attachments(
            second_state.session_id, second_state.run_id
        ) == (second_fact,)
        assert await reopened.artifact_has_active_attachments(shared.artifact_id)

    asyncio.run(scenario())


def test_concurrent_detach_is_single_commit_and_retry_is_idempotent() -> None:
    async def scenario() -> None:
        journal = InMemoryRunJournal()
        batch = acceptance("concurrent-lifecycle")
        terminal = replace(batch.state, status=RunStatus.COMPLETED, final_text="done")
        fact = ArtifactAttachmentFact(
            terminal.session_id,
            terminal.run_id,
            terminal.submission_id,
            "owner",
            ArtifactAttachmentRole.FINAL_OUTPUT,
            ArtifactAttachmentAction.ATTACHED,
            artifact_ref("f"),
        )
        await journal.commit(replace(batch, state=terminal, artifact_facts=(fact,)), 0)
        first = ArtifactLifecycleService(journal=journal)
        second = ArtifactLifecycleService(journal=journal)
        results = await asyncio.gather(
            first.detach(terminal.session_id, terminal.run_id),
            second.detach(terminal.session_id, terminal.run_id),
        )
        assert sorted(len(result) for result in results) == [0, 1]
        history = await journal.load_artifact_facts(terminal.session_id, terminal.run_id, -1, 16)
        assert [record.fact.action for record in history].count(
            ArtifactAttachmentAction.DETACHED
        ) == 1

    asyncio.run(scenario())


def test_stale_cas_is_reloaded_before_detach_retry() -> None:
    class ConflictOnceJournal(InMemoryRunJournal):
        fail_next = False

        async def commit(self, batch, expected_version):
            if self.fail_next:
                self.fail_next = False
                await super().commit(JournalCommit(batch.state), expected_version)
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            return await super().commit(batch, expected_version)

    async def scenario() -> None:
        journal = ConflictOnceJournal()
        batch = acceptance("stale-lifecycle")
        terminal = replace(batch.state, status=RunStatus.COMPLETED, final_text="done")
        fact = ArtifactAttachmentFact(
            terminal.session_id,
            terminal.run_id,
            terminal.submission_id,
            "owner",
            ArtifactAttachmentRole.FINAL_OUTPUT,
            ArtifactAttachmentAction.ATTACHED,
            artifact_ref("9"),
        )
        await journal.commit(replace(batch, state=terminal, artifact_facts=(fact,)), 0)
        journal.fail_next = True
        detached = await ArtifactLifecycleService(journal=journal).detach(
            terminal.session_id, terminal.run_id
        )
        assert detached == (replace(fact, action=ArtifactAttachmentAction.DETACHED),)

    asyncio.run(scenario())
