import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.artifact_lifecycle import ArtifactLifecycleService
from src.application.recovery import RecoveryCorruption, RecoveryService
from src.core.artifacts import ArtifactRef, ArtifactRepository, ExternalizedContent
from src.core.commands import RunLimits
from src.core.errors import RunErrorCode
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ConversationSummaryCheckpoint,
    JournalCommit,
)
from src.core.state import RunState, RunStatus
from tests.unit.core.test_journal_contract import acceptance
from tests.unit.core.test_run_state import (
    advance,
    ask_approval,
    begin_step,
    decide,
    finish_tool,
    initial,
    resume,
    submit,
)

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)


class FixedClock:
    def now(self):
        return NOW

    def monotonic(self):
        return 0.0


class RecoveryIds:
    def __init__(self):
        self.index = 0

    def event(self):
        self.index += 1
        return EventId(f"recovery-{self.index}")


def service(path: Path):
    journal = SQLiteRunJournal(path.resolve())
    return journal, RecoveryService(
        journal=journal,
        artifacts=InMemoryArtifactRepository(1024),
        clock=FixedClock(),
        ids=RecoveryIds(),
    )


def active_fact(
    state: RunState,
    artifact: ArtifactRef,
    role: ArtifactAttachmentRole,
    owner_id: str,
) -> ArtifactAttachmentFact:
    return ArtifactAttachmentFact(
        state.session_id,
        state.run_id,
        state.submission_id,
        owner_id,
        role,
        ArtifactAttachmentAction.ATTACHED,
        artifact,
    )


def test_recovery_closes_running_state_without_replaying_work(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "recover.sqlite3")
    batch = replace(acceptance("open"), summary=None)
    asyncio.run(journal.commit(batch, 0))

    outcome = asyncio.run(recovery.recover_session(batch.state.session_id))

    assert outcome.changed and outcome.state is not None
    assert outcome.state.status is RunStatus.FAILED
    assert outcome.state.failure is not None
    assert outcome.state.failure.code is RunErrorCode.MODEL_FAILED
    again = asyncio.run(recovery.recover_session(batch.state.session_id))
    assert not again.changed and again.state == outcome.state
    snapshot = asyncio.run(journal.load_session(batch.state.session_id))
    assert [event.event.type.value for event in snapshot.events].count("run.failed") == 1


def test_recovery_preserves_valid_waiting_approval(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "paused.sqlite3")
    waiting = advance(initial(), submit(), *begin_step(), ask_approval())
    waiting = replace(waiting, journal_version=1, journal_sequence=0)
    asyncio.run(journal.commit(JournalCommit(waiting), 0))

    outcome = asyncio.run(recovery.recover_session(waiting.session_id))

    assert not outcome.changed and outcome.state is not None
    assert outcome.state.status is RunStatus.WAITING_APPROVAL
    assert asyncio.run(journal.load_head(waiting.session_id)).version == 1


@pytest.mark.parametrize("approved", [True, False])
def test_recovery_preserves_durable_decision_for_runtime_resume(
    tmp_path: Path, approved: bool
) -> None:
    journal, recovery = service(tmp_path / f"decided-{approved}.sqlite3")
    decided = advance(initial(), submit(), *begin_step(), ask_approval(), decide(approved))
    decided = replace(decided, journal_version=1, journal_sequence=0)
    asyncio.run(journal.commit(JournalCommit(decided), 0))

    outcome = asyncio.run(recovery.recover_session(decided.session_id))

    assert not outcome.changed
    assert outcome.state == decided
    assert outcome.state.status is RunStatus.WAITING_APPROVAL


def test_recovery_settles_durable_unexecuted_tool_batch_without_gateway(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "tool-batch.sqlite3")
    pending = advance(initial(), submit(), *begin_step())
    pending = replace(pending, journal_version=1, journal_sequence=0)
    asyncio.run(journal.commit(JournalCommit(pending), 0))

    outcome = asyncio.run(recovery.recover_session(pending.session_id))

    assert outcome.changed and outcome.state is not None
    assert outcome.state.status is RunStatus.FAILED
    assert outcome.state.failure is not None
    assert outcome.state.failure.code is RunErrorCode.TOOL_FAILED


def test_recovery_terminal_is_idempotent(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "terminal.sqlite3")
    state = RunState(
        RunId("terminal-run"),
        SessionId("terminal-session"),
        SubmissionId("terminal-submit"),
        ProfileId("profile"),
        RunLimits(2, 10, 128),
        version=2,
        status=RunStatus.COMPLETED,
        final_text="done",
        journal_version=1,
    )
    asyncio.run(journal.commit(JournalCommit(state), 0))
    first = asyncio.run(recovery.recover_session(state.session_id))
    second = asyncio.run(recovery.recover_session(state.session_id))
    assert first == second
    assert not first.changed


def test_terminal_dangling_artifact_is_repository_corruption(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "corrupt.sqlite3")
    batch = acceptance("artifact")
    assert batch.summary is not None
    terminal = replace(
        batch.state,
        version=2,
        status=RunStatus.COMPLETED,
        final_text="summary",
        final_artifacts=batch.summary.artifacts,
        journal_sequence=0,
    )
    fact = active_fact(
        terminal, terminal.final_artifacts[0], ArtifactAttachmentRole.FINAL_OUTPUT, "final"
    )
    asyncio.run(journal.commit(JournalCommit(terminal, artifact_facts=(fact,)), 0))

    with pytest.raises(RecoveryCorruption) as caught:
        asyncio.run(recovery.recover_session(terminal.session_id))
    assert caught.value.failure.code is RunErrorCode.ARTIFACT_FAILED


def test_detached_terminal_final_ref_is_not_recovery_corruption(tmp_path: Path) -> None:
    path = (tmp_path / "detached-final.sqlite3").resolve()
    journal = SQLiteRunJournal(path)
    batch = acceptance("detached-final")
    assert batch.summary is not None
    ref = batch.summary.artifacts[0]
    terminal = replace(
        batch.state,
        version=2,
        status=RunStatus.COMPLETED,
        final_text="summary",
        final_artifacts=(ref,),
    )
    fact = ArtifactAttachmentFact(
        terminal.session_id,
        terminal.run_id,
        terminal.submission_id,
        "final",
        ArtifactAttachmentRole.FINAL_OUTPUT,
        ArtifactAttachmentAction.ATTACHED,
        ref,
    )
    asyncio.run(journal.commit(replace(batch, state=terminal, artifact_facts=(fact,)), 0))
    asyncio.run(
        ArtifactLifecycleService(journal=journal).detach(terminal.session_id, terminal.run_id)
    )
    reopened = SQLiteRunJournal(path)
    recovery = RecoveryService(
        journal=reopened,
        artifacts=InMemoryArtifactRepository(1024),
        clock=FixedClock(),
        ids=RecoveryIds(),
    )
    outcome = asyncio.run(recovery.recover_session(terminal.session_id))
    assert outcome.state is not None and outcome.state.status is RunStatus.COMPLETED
    assert not outcome.changed


def test_completed_tool_attachment_survives_state_cleanup_and_restart(tmp_path: Path) -> None:
    path = (tmp_path / "tool-attachment.sqlite3").resolve()
    journal = SQLiteRunJournal(path)
    batch = acceptance("settled-tool")
    assert batch.summary is not None
    ref = batch.summary.artifacts[0]
    terminal = replace(
        batch.state,
        version=2,
        status=RunStatus.COMPLETED,
        final_text="done",
        final_artifacts=(),
        current_step=None,
    )
    fact = ArtifactAttachmentFact(
        terminal.session_id,
        terminal.run_id,
        terminal.submission_id,
        "tool-call-1",
        ArtifactAttachmentRole.TOOL_RESULT,
        ArtifactAttachmentAction.ATTACHED,
        ref,
    )
    asyncio.run(journal.commit(replace(batch, state=terminal, artifact_facts=(fact,)), 0))

    reopened = SQLiteRunJournal(path)
    recovery = RecoveryService(
        journal=reopened,
        artifacts=InMemoryArtifactRepository(1024),
        clock=FixedClock(),
        ids=RecoveryIds(),
    )
    with pytest.raises(RecoveryCorruption) as caught:
        asyncio.run(recovery.recover_session(terminal.session_id))
    assert caught.value.failure.code is RunErrorCode.ARTIFACT_FAILED


def test_active_dangling_artifact_settles_once_then_is_idempotent(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "active-corrupt.sqlite3")
    batch = acceptance("artifact-active")
    assert batch.summary is not None
    fact = active_fact(
        batch.state,
        batch.summary.artifacts[0],
        ArtifactAttachmentRole.SUMMARY,
        "summary",
    )
    asyncio.run(journal.commit(replace(batch, artifact_facts=(fact,)), 0))

    first = asyncio.run(recovery.recover_session(batch.state.session_id))
    second = asyncio.run(recovery.recover_session(batch.state.session_id))

    assert first.changed and first.state is not None
    assert first.state.failure is not None
    assert first.state.failure.code is RunErrorCode.ARTIFACT_FAILED
    assert second == replace(first, changed=False)
    snapshot = asyncio.run(journal.load_session(batch.state.session_id))
    assert [event.event.type.value for event in snapshot.events].count("run.failed") == 1


@pytest.mark.parametrize("mode", ["object", "raises", "wrong", "tampered"])
def test_untrusted_artifact_stat_maps_to_stable_corruption(tmp_path: Path, mode: str) -> None:
    journal = SQLiteRunJournal((tmp_path / f"stat-{mode}.sqlite3").resolve())
    batch = acceptance(f"stat-{mode}")
    assert batch.summary is not None
    terminal = replace(
        batch.state,
        version=2,
        status=RunStatus.COMPLETED,
        final_text="summary",
        final_artifacts=batch.summary.artifacts,
        journal_sequence=0,
    )
    fact = active_fact(
        terminal, terminal.final_artifacts[0], ArtifactAttachmentRole.FINAL_OUTPUT, "final"
    )
    asyncio.run(journal.commit(JournalCommit(terminal, artifact_facts=(fact,)), 0))
    expected = terminal.final_artifacts[0]

    class BadRepository:
        async def stat(self, artifact_id: str):
            if mode == "raises":
                raise RuntimeError("host detail")
            if mode == "object":
                return object()
            if mode == "wrong":
                return ArtifactRef(
                    expected.artifact_id,
                    "text/plain",
                    expected.size_bytes,
                    expected.sha256,
                )
            result = ArtifactRef(
                expected.artifact_id,
                expected.media_type,
                expected.size_bytes,
                expected.sha256,
            )
            object.__setattr__(result, "size_bytes", 99)
            return result

    recovery = RecoveryService(
        journal=journal,
        artifacts=cast(ArtifactRepository, BadRepository()),
        clock=FixedClock(),
        ids=RecoveryIds(),
    )
    with pytest.raises(RecoveryCorruption) as caught:
        asyncio.run(recovery.recover_session(terminal.session_id))
    assert caught.value.failure.code is RunErrorCode.ARTIFACT_FAILED


def _conflicting_ref(ref: ArtifactRef) -> ArtifactRef:
    return ArtifactRef(ref.artifact_id, "text/plain", ref.size_bytes, ref.sha256)


class _OneDescriptorRepository:
    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        self.stat_calls = 0

    async def stat(self, artifact_id: str):
        self.stat_calls += 1
        return self.ref if artifact_id == self.ref.artifact_id else None


def test_final_and_summary_same_id_with_conflicting_descriptors_is_stable_corruption(
    tmp_path: Path,
) -> None:
    journal = SQLiteRunJournal((tmp_path / "final-summary-conflict.sqlite3").resolve())
    batch = acceptance("descriptor-conflict")
    assert batch.summary is not None
    expected = batch.summary.artifacts[0]
    terminal = replace(
        batch.state,
        version=2,
        status=RunStatus.COMPLETED,
        final_text="done",
        final_artifacts=(_conflicting_ref(expected),),
    )
    facts = (
        active_fact(
            terminal,
            terminal.final_artifacts[0],
            ArtifactAttachmentRole.FINAL_OUTPUT,
            "final",
        ),
        active_fact(terminal, expected, ArtifactAttachmentRole.SUMMARY, "summary"),
    )
    asyncio.run(journal.commit(replace(batch, state=terminal, artifact_facts=facts), 0))
    repository = _OneDescriptorRepository(terminal.final_artifacts[0])
    recovery = RecoveryService(
        journal=journal,
        artifacts=cast(ArtifactRepository, repository),
        clock=FixedClock(),
        ids=RecoveryIds(),
    )

    for _ in range(2):
        with pytest.raises(RecoveryCorruption) as caught:
            asyncio.run(recovery.recover_session(terminal.session_id))
        assert caught.value.failure.code is RunErrorCode.ARTIFACT_FAILED
    assert repository.stat_calls == 0


def test_tool_and_summary_same_id_with_conflicting_descriptors_settles_once(
    tmp_path: Path,
) -> None:
    journal = SQLiteRunJournal((tmp_path / "tool-summary-conflict.sqlite3").resolve())
    template = acceptance("tool-descriptor")
    assert template.summary is not None
    expected = template.summary.artifacts[0]
    state = advance(
        initial(), submit(), *begin_step(), ask_approval(), decide(), resume(), finish_tool()
    )
    step = state.current_step
    assert step is not None and step.tool_calls
    tool_ref = _conflicting_ref(expected)
    tool = replace(
        step.tool_calls[0],
        result=ExternalizedContent(artifacts=(tool_ref,), summary="tool result"),
    )
    state = replace(
        state,
        current_step=replace(step, tool_calls=(tool,)),
        journal_version=1,
        journal_sequence=1,
    )
    event = DomainEvent(
        EventId("tool-conflict-event"),
        DomainEventType.RUN_STARTED,
        state.run_id,
        state.session_id,
        NOW,
        {},
    )
    summary = ConversationSummaryCheckpoint(
        state.session_id,
        1,
        "v1",
        state.profile_id,
        "summary",
        artifacts=(expected,),
    )
    facts = (
        active_fact(state, tool_ref, ArtifactAttachmentRole.TOOL_RESULT, "tool"),
        active_fact(state, expected, ArtifactAttachmentRole.SUMMARY, "summary"),
    )
    asyncio.run(
        journal.commit(
            JournalCommit(state, events=(event,), summary=summary, artifact_facts=facts), 0
        )
    )
    repository = _OneDescriptorRepository(tool_ref)
    recovery = RecoveryService(
        journal=journal,
        artifacts=cast(ArtifactRepository, repository),
        clock=FixedClock(),
        ids=RecoveryIds(),
    )

    first = asyncio.run(recovery.recover_session(state.session_id))
    second = asyncio.run(recovery.recover_session(state.session_id))

    assert first.changed and first.state is not None and first.state.failure is not None
    assert first.state.failure.code is RunErrorCode.ARTIFACT_FAILED
    assert second == replace(first, changed=False)
    assert repository.stat_calls == 0
