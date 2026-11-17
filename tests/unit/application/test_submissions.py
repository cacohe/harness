import asyncio
from datetime import datetime, timedelta, timezone
from itertools import count

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.submissions import AcceptedRun, SubmissionService
from src.core.commands import CommandContext, CompleteRun, RunLimits
from src.core.errors import InvalidTransition, RunErrorCode
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import JournalCommit, JournalError, JournalSnapshot, RunJournal
from src.core.state import RunStatus, TaskStateMachine

SESSION = SessionId("session")
FIRST = SubmissionId("first")
SECOND = SubmissionId("second")
PROFILE = ProfileId("default")
NOW = datetime(2026, 11, 17, 1, 30, tzinfo=timezone.utc)
LIMITS = RunLimits(5, 30, 1024)


class SequenceIds:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.runs = count(1)
        self.events = count(1)

    def run(self) -> RunId:
        return RunId(f"{self.prefix}-run-{next(self.runs)}")

    def event(self) -> EventId:
        return EventId(f"{self.prefix}-event-{next(self.events)}")

    def session(self) -> SessionId:
        raise AssertionError("the caller supplies the session ID")

    def submission(self) -> SubmissionId:
        raise AssertionError("the caller supplies the submission ID")

    def profile(self) -> ProfileId:
        raise AssertionError("the caller supplies the profile ID")


class FixedClock:
    def __init__(self, timestamp: datetime) -> None:
        self.timestamp = timestamp

    def now(self) -> datetime:
        return self.timestamp

    def monotonic(self) -> float:
        return 42.0


def service(
    journal: RunJournal,
    prefix: str = "service",
    *,
    timestamp: datetime = NOW,
    limits: RunLimits = LIMITS,
) -> SubmissionService:
    return SubmissionService(
        journal=journal, ids=SequenceIds(prefix), clock=FixedClock(timestamp), limits=limits
    )


async def finish(journal: RunJournal, accepted: AcceptedRun) -> None:
    state = accepted.state
    command = CompleteRun(
        state.run_id,
        state.session_id,
        "done",
        context=CommandContext(NOW, 1, (EventId("completed"),)),
    )
    terminal = TaskStateMachine.apply(state, command)
    await journal.commit(JournalCommit(terminal.state, terminal.events), accepted.version)


def test_accept_persists_running_state_start_event_and_exact_user_text() -> None:
    journal = InMemoryRunJournal()
    message = "  caf\u00e9\n"

    accepted = asyncio.run(service(journal).accept(SESSION, message, FIRST, PROFILE))

    snapshot = asyncio.run(journal.load_session(SESSION))
    receipt = asyncio.run(journal.find_submission(SESSION, FIRST))
    assert accepted.state.status is RunStatus.RUNNING
    assert (accepted.version, accepted.state.version, snapshot.sequence) == (1, 1, 2)
    assert snapshot.state == accepted.state
    assert accepted.state.run_id == RunId("service-run-1")
    assert accepted.state.limits == LIMITS
    assert receipt is not None and receipt.state == accepted.state and receipt.version == 1
    assert (
        receipt.submission.digest
        == "81769635077a631fc77dae568dcb7e0fdc220e03fba4b4fd48aa7afe89c460bb"
    )
    assert receipt.submission.run_id == accepted.state.run_id
    (event,) = snapshot.events
    assert event.sequence == 1
    assert event.event == DomainEvent(
        EventId("service-event-1"),
        DomainEventType.RUN_STARTED,
        accepted.state.run_id,
        SESSION,
        NOW,
        {},
    )
    (entry,) = snapshot.transcript
    assert (entry.sequence, entry.role, entry.content) == (2, "user", message)
    assert (entry.session_id, entry.run_id, entry.submission_id) == (
        SESSION,
        accepted.state.run_id,
        FIRST,
    )


def test_recreated_service_replays_original_receipt_after_next_run_starts() -> None:
    async def scenario() -> None:
        journal = InMemoryRunJournal()
        original = await service(journal, "original").accept(SESSION, "hello", FIRST, PROFILE)
        await finish(journal, original)
        recreated = service(
            journal, "recreated", timestamp=NOW + timedelta(hours=1), limits=RunLimits(2, 10, 256)
        )
        next_run = await recreated.accept(SESSION, "next", SECOND, PROFILE)
        assert (next_run.version, next_run.state.version) == (3, 1)
        assert next_run.state.run_id != original.state.run_id
        assert next_run.state.status is RunStatus.RUNNING
        before = await journal.load_session(SESSION)

        assert await recreated.accept(SESSION, "hello", FIRST, PROFILE) == original
        assert await journal.load_session(SESSION) == before

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("message", "profile"),
    [("hello ", PROFILE), ("hello", ProfileId("other"))],
)
def test_reusing_submission_id_with_changed_intent_conflicts(
    message: str, profile: ProfileId
) -> None:
    journal = InMemoryRunJournal()
    submissions = service(journal)
    asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))
    before = asyncio.run(journal.load_session(SESSION))

    with pytest.raises(JournalError) as error:
        asyncio.run(submissions.accept(SESSION, message, FIRST, profile))

    assert error.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
    assert asyncio.run(journal.load_session(SESSION)) == before


def test_new_submission_is_rejected_while_run_is_active() -> None:
    journal = InMemoryRunJournal()
    submissions = service(journal)
    asyncio.run(submissions.accept(SESSION, "first", FIRST, PROFILE))
    before = asyncio.run(journal.load_session(SESSION))

    with pytest.raises(JournalError) as error:
        asyncio.run(submissions.accept(SESSION, "second", SECOND, PROFILE))

    assert error.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
    assert asyncio.run(journal.load_session(SESSION)) == before
    assert asyncio.run(journal.find_submission(SESSION, SECOND)) is None


class CompetingJournal(InMemoryRunJournal):
    def synchronize_next_two_commits(self) -> None:
        self.barrier = asyncio.Barrier(2)

    barrier: asyncio.Barrier | None = None

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        barrier = self.barrier
        if barrier is not None:
            await asyncio.wait_for(barrier.wait(), timeout=5)
            self.barrier = None
        return await super().commit(batch, expected_version)


@pytest.mark.parametrize("starts_terminal", [False, True])
@pytest.mark.parametrize("second_submission", [FIRST, SECOND])
def test_competing_submissions_resolve_one_atomic_cas(
    starts_terminal: bool, second_submission: SubmissionId
) -> None:
    async def scenario() -> None:
        journal = CompetingJournal()
        if starts_terminal:
            original = await service(journal, "previous").accept(
                SESSION, "previous", SubmissionId("previous"), PROFILE
            )
            await finish(journal, original)
        before = await journal.load_session(SESSION)
        journal.synchronize_next_two_commits()

        first, second = await asyncio.gather(
            service(journal, "one").accept(SESSION, "hello", FIRST, PROFILE),
            service(journal, "two").accept(SESSION, "hello", second_submission, PROFILE),
            return_exceptions=True,
        )

        snapshot = await journal.load_session(SESSION, after_sequence=before.sequence)
        assert snapshot.version == before.version + 1
        assert snapshot.sequence == before.sequence + 2
        assert len(snapshot.events) == len(snapshot.transcript) == 1
        if second_submission == FIRST:
            assert isinstance(first, AcceptedRun)
            assert second == first
            assert snapshot.state == first.state
        else:
            winner, loser = (first, second) if isinstance(first, AcceptedRun) else (second, first)
            assert isinstance(winner, AcceptedRun)
            assert isinstance(loser, JournalError)
            assert loser.failure.code is RunErrorCode.JOURNAL_CONFLICT
            assert snapshot.state == winner.state
            loser_id = SECOND if winner.state.submission_id == FIRST else FIRST
            assert await journal.find_submission(SESSION, loser_id) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "timestamp",
    [NOW.replace(fold=1), NOW.replace(tzinfo=timezone(timedelta(hours=1)))],
)
def test_acceptance_does_not_relax_domain_timestamp_rules(timestamp: datetime) -> None:
    journal = InMemoryRunJournal()
    with pytest.raises(InvalidTransition):
        asyncio.run(service(journal, timestamp=timestamp).accept(SESSION, "hello", FIRST, PROFILE))

    assert asyncio.run(journal.load_session(SESSION)).version == 0
    assert asyncio.run(journal.find_submission(SESSION, FIRST)) is None
