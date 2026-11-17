import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.recovery import RecoveryCorruption, RecoveryService
from src.core.artifacts import ArtifactRef, ArtifactRepository
from src.core.commands import RunLimits
from src.core.errors import RunErrorCode
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import JournalCommit
from src.core.state import RunState, RunStatus
from tests.unit.core.test_journal_contract import acceptance
from tests.unit.core.test_run_state import advance, ask_approval, begin_step, initial, submit

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


def test_recovery_settles_durable_unexecuted_tool_batch_without_gateway(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "tool-batch.sqlite3")
    pending = advance(initial(), submit(), *begin_step())
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
    )
    asyncio.run(journal.commit(JournalCommit(terminal), 0))

    with pytest.raises(RecoveryCorruption) as caught:
        asyncio.run(recovery.recover_session(terminal.session_id))
    assert caught.value.failure.code is RunErrorCode.ARTIFACT_FAILED


def test_active_dangling_artifact_settles_once_then_is_idempotent(tmp_path: Path) -> None:
    journal, recovery = service(tmp_path / "active-corrupt.sqlite3")
    batch = acceptance("artifact-active")
    asyncio.run(journal.commit(batch, 0))

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
    )
    asyncio.run(journal.commit(JournalCommit(terminal), 0))
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
