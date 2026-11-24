import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import count
from typing import cast

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.submissions import AcceptedRun, SubmissionService
from src.core.execution.commands import (
    CommandContext,
    CompleteRun,
    FinishModelCall,
    RunLimits,
    StartModelCall,
    StartStep,
    StartToolCall,
    ToolCallRequest,
)
from src.core.execution.errors import InvalidTransition, RunErrorCode
from src.core.execution.events import DomainEvent, DomainEventType
from src.core.execution.state import RunStatus, TaskStateMachine
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.persistence.journal import (
    JournalCommit,
    JournalError,
    JournalHead,
    JournalSnapshot,
    RunJournal,
    SubmissionAcceptance,
)

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
        self.readings = 0

    def now(self) -> datetime:
        self.readings += 1
        return self.timestamp

    def monotonic(self) -> float:
        self.readings += 1
        return 42.0


def service(
    journal: RunJournal,
    prefix: str = "service",
    *,
    timestamp: datetime = NOW,
    limits: RunLimits = LIMITS,
    max_user_message_bytes: int = 1_048_576,
) -> SubmissionService:
    return SubmissionService(
        journal=journal,
        ids=SequenceIds(prefix),
        clock=FixedClock(timestamp),
        limits=limits,
        max_user_message_bytes=max_user_message_bytes,
    )


def test_message_utf8_limit_is_checked_before_canonical_json(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("oversized input reached json.dumps")

    monkeypatch.setattr("src.application.submissions.json.dumps", forbidden)
    with pytest.raises(InvalidTransition) as error:
        asyncio.run(
            service(InMemoryRunJournal(), max_user_message_bytes=8).accept(
                SESSION, "😀😀😀", FIRST, PROFILE
            )
        )
    assert error.value.failure.code is RunErrorCode.LIMIT_EXCEEDED


def test_message_utf8_limit_accepts_exact_multibyte_boundary() -> None:
    accepted = asyncio.run(
        service(InMemoryRunJournal(), max_user_message_bytes=8).accept(
            SESSION, "😀😀", FIRST, PROFILE
        )
    )
    assert accepted.state.status is RunStatus.RUNNING


@pytest.mark.parametrize("limit", [True, 0, 64 * 1_048_576 + 1])
def test_message_limit_rejects_invalid_numeric_values(limit: object) -> None:
    with pytest.raises(ValueError):
        SubmissionService(
            InMemoryRunJournal(),
            SequenceIds("invalid"),
            FixedClock(NOW),
            LIMITS,
            cast(int, limit),
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
    persisted = replace(
        terminal.state,
        journal_version=accepted.version + 1,
        journal_sequence=state.journal_sequence + len(terminal.events),
    )
    await journal.commit(JournalCommit(persisted, terminal.events), accepted.version)


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


def test_waiting_for_approval_reserves_the_session() -> None:
    async def scenario() -> None:
        journal = InMemoryRunJournal()
        submissions = service(journal)
        accepted = await submissions.accept(SESSION, "first", FIRST, PROFILE)
        state = accepted.state
        identity = (state.run_id, SESSION)
        context = CommandContext(NOW, 1, (EventId("waiting"),))
        request = ToolCallRequest(
            "tool-call",
            "test-tool",
            "{}",
            requires_approval=True,
            approval_scope="test-scope",
            workspace_id="test-workspace",
        )
        transition = TaskStateMachine.apply(state, StartStep(*identity, "step", context=context))
        transition = TaskStateMachine.apply(
            transition.state, StartModelCall(*identity, "step", "model", context=context)
        )
        transition = TaskStateMachine.apply(
            transition.state,
            FinishModelCall(*identity, "step", "model", "", (request,), context=context),
        )
        transition = TaskStateMachine.apply(
            transition.state,
            StartToolCall(*identity, "step", "model", "tool-call", True, context=context),
        )
        assert transition.state.status is RunStatus.WAITING_APPROVAL
        persisted = replace(
            transition.state,
            journal_version=accepted.version + 1,
            journal_sequence=state.journal_sequence + len(transition.events),
        )
        await journal.commit(JournalCommit(persisted, transition.events), accepted.version)
        before = await journal.load_session(SESSION)

        with pytest.raises(JournalError) as error:
            await submissions.accept(SESSION, "next", SECOND, PROFILE)

        assert error.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
        assert await journal.load_session(SESSION) == before
        assert await journal.find_submission(SESSION, SECOND) is None

    asyncio.run(scenario())


class HeadOnlyJournal(InMemoryRunJournal):
    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        pytest.fail("acceptance must not read Session history")


def test_initial_acceptance_and_replay_need_only_the_head_and_receipt() -> None:
    journal = HeadOnlyJournal()
    original = asyncio.run(service(journal).accept(SESSION, "hello", FIRST, PROFILE))
    ids = SequenceIds("recreated")
    clock = FixedClock(NOW)
    recreated = SubmissionService(journal, ids, clock, LIMITS)

    assert asyncio.run(recreated.accept(SESSION, "hello", FIRST, PROFILE)) == original
    assert (next(ids.runs), next(ids.events), clock.readings) == (1, 1, 0)


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("session", "message", "submission", "profile"),
    [
        (RunId("session"), "hello", FIRST, PROFILE),
        (SESSION, StringSubclass("hello"), FIRST, PROFILE),
        (SESSION, "\ud800", FIRST, PROFILE),
        (SESSION, "hello", RunId("first"), PROFILE),
        (SESSION, "hello", FIRST, RunId("default")),
        (SESSION, "hello", FIRST, None),
    ],
)
def test_invalid_input_is_rejected_before_using_ports(
    session: object, message: object, submission: object, profile: object
) -> None:
    with pytest.raises(InvalidTransition) as error:
        asyncio.run(
            service(cast(RunJournal, object())).accept(
                cast(SessionId, session),
                cast(str, message),
                cast(SubmissionId, submission),
                cast(ProfileId, profile),
            )
        )
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


class CorruptReceiptJournal(InMemoryRunJournal):
    corrupt: Callable[[SubmissionAcceptance], SubmissionAcceptance] | None = None
    head_reads = 0

    async def load_head(self, session_id: SessionId) -> JournalHead:
        self.head_reads += 1
        return await super().load_head(session_id)

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionAcceptance | None:
        receipt = await super().find_submission(session_id, submission_id)
        return (
            self.corrupt(receipt) if receipt is not None and self.corrupt is not None else receipt
        )


def corrupt_identity_value(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
    object.__setattr__(receipt.state.run_id, "value", StringSubclass(receipt.state.run_id.value))
    return receipt


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda r: replace(r, submission=replace(r.submission, session_id=SessionId("other"))),
        lambda r: replace(r, submission=replace(r.submission, submission_id=SECOND)),
        lambda r: replace(r, submission=replace(r.submission, run_id=RunId("other"))),
        lambda r: replace(r, state=replace(r.state, session_id=SessionId("other"))),
        lambda r: replace(r, state=replace(r.state, submission_id=SECOND)),
        lambda r: replace(r, state=replace(r.state, profile_id=ProfileId("other"))),
        lambda r: replace(r, state=replace(r.state, status=RunStatus.CREATED)),
        lambda r: replace(r, state=replace(r.state, version=2)),
        lambda r: replace(r, state=replace(r.state, elapsed_seconds=1)),
        lambda r: replace(r, state=replace(r.state, limits=corrupt_limits())),
        lambda r: replace(r, version=True),
        lambda r: replace(r, version=0),
        corrupt_identity_value,
    ],
)
def test_replay_rejects_corrupt_acceptance(
    corrupt: Callable[[SubmissionAcceptance], SubmissionAcceptance],
) -> None:
    journal = CorruptReceiptJournal()
    submissions = service(journal)
    asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))
    before = asyncio.run(journal.load_session(SESSION))
    journal.corrupt = corrupt

    with pytest.raises(JournalError) as error:
        asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert asyncio.run(journal.load_session(SESSION)) == before


def corrupt_limits() -> RunLimits:
    limits = replace(LIMITS)
    object.__setattr__(limits, "max_steps", 0)
    return limits


class IntSubclass(int):
    pass


class ExplosiveFloat(float):
    def __le__(self, other: object) -> bool:
        raise RuntimeError("overloaded numeric comparison")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", IntSubclass(LIMITS.max_steps)),
        ("max_inline_result_bytes", IntSubclass(LIMITS.max_inline_result_bytes)),
        ("max_elapsed_seconds", ExplosiveFloat(LIMITS.max_elapsed_seconds)),
        ("max_elapsed_seconds", float("inf")),
        ("max_elapsed_seconds", True),
    ],
)
def test_replay_rejects_corrupt_limit_primitives(field: str, value: object) -> None:
    journal = CorruptReceiptJournal()
    submissions = service(journal)
    asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))

    def corrupt(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
        object.__setattr__(receipt.state.limits, field, value)
        return receipt

    journal.corrupt = corrupt
    with pytest.raises(JournalError) as error:
        asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def future_version(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
    return replace(receipt, version=999)


def other_state(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
    return replace(receipt, state=replace(receipt.state, limits=RunLimits(3, 10, 64)))


def other_elapsed_type(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
    limits = replace(
        receipt.state.limits,
        max_elapsed_seconds=float(receipt.state.limits.max_elapsed_seconds),
    )
    return replace(receipt, state=replace(receipt.state, limits=limits))


def other_elapsed_zero(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
    return replace(receipt, state=replace(receipt.state, elapsed_seconds=0.0))


@pytest.mark.parametrize(
    "corrupt", [future_version, other_state, other_elapsed_type, other_elapsed_zero]
)
def test_replay_cross_checks_receipt_against_session_head(
    corrupt: Callable[[SubmissionAcceptance], SubmissionAcceptance],
) -> None:
    journal = CorruptReceiptJournal()
    submissions = service(journal)
    asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))
    journal.corrupt = corrupt
    before = journal.head_reads

    with pytest.raises(JournalError) as error:
        asyncio.run(submissions.accept(SESSION, "hello", FIRST, PROFILE))

    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION
    assert journal.head_reads - before == (2 if corrupt is future_version else 1)


class InterleavedJournal(CorruptReceiptJournal):
    after_next_head: Callable[[], Awaitable[None]] | None = None

    async def load_head(self, session_id: SessionId) -> JournalHead:
        head = await super().load_head(session_id)
        if self.after_next_head is not None:
            action, self.after_next_head = self.after_next_head, None
            await action()
        return head


@pytest.mark.parametrize("finish_before_lookup", [False, True])
def test_replay_confirms_acceptance_committed_after_first_head(
    finish_before_lookup: bool,
) -> None:
    async def scenario() -> None:
        journal = InterleavedJournal()
        winners: list[AcceptedRun] = []

        async def concurrent_acceptance() -> None:
            winner = await service(journal, "winner").accept(SESSION, "hello", FIRST, PROFILE)
            winners.append(winner)
            if finish_before_lookup:
                await finish(journal, winner)

        journal.after_next_head = concurrent_acceptance
        ids = SequenceIds("replay")
        clock = FixedClock(NOW)
        replay = SubmissionService(journal, ids, clock, LIMITS)

        accepted = await replay.accept(SESSION, "hello", FIRST, PROFILE)

        assert winners == [accepted]
        # The initial read, the winning request's read, and one confirming read.
        assert journal.head_reads == 3
        assert (next(ids.runs), next(ids.events), clock.readings) == (1, 1, 0)

    asyncio.run(scenario())


class CompetingJournal(CorruptReceiptJournal):
    def synchronize_next_two_commits(self) -> None:
        self.barrier = asyncio.Barrier(2)

    barrier: asyncio.Barrier | None = None

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        barrier = self.barrier
        if barrier is not None:
            await asyncio.wait_for(barrier.wait(), timeout=5)
            self.barrier = None
        return await super().commit(batch, expected_version)


@pytest.mark.parametrize("field", ["elapsed_seconds", "max_elapsed_seconds"])
@pytest.mark.parametrize("cas_loser", [False, True])
def test_replay_rejects_numbers_with_overloaded_metaclass_equality(
    field: str, cas_loser: bool
) -> None:
    comparisons: list[str] = []

    class ForgedNumberType(type):
        def __eq__(cls, other: object) -> bool:
            comparisons.append("type equality")
            return True

    class ForgedFloat(float, metaclass=ForgedNumberType):
        def __ne__(self, other: object) -> bool:
            raise RuntimeError("elapsed value comparison")

        def __le__(self, other: object) -> bool:
            raise RuntimeError("elapsed limit comparison")

    def corrupt(receipt: SubmissionAcceptance) -> SubmissionAcceptance:
        if field == "elapsed_seconds":
            object.__setattr__(receipt.state, field, ForgedFloat(0))
        else:
            object.__setattr__(receipt.state.limits, field, ForgedFloat(30))
        return receipt

    async def scenario() -> None:
        journal = CompetingJournal()
        journal.corrupt = corrupt
        first = service(journal, "first")
        if cas_loser:
            journal.synchronize_next_two_commits()
            results = await asyncio.gather(
                first.accept(SESSION, "hello", FIRST, PROFILE),
                service(journal, "second").accept(SESSION, "hello", FIRST, PROFILE),
                return_exceptions=True,
            )
            accepted = [result for result in results if isinstance(result, AcceptedRun)]
            assert len(accepted) == 1
            (error,) = [result for result in results if result is not accepted[0]]
            assert isinstance(error, JournalError)
        else:
            await first.accept(SESSION, "hello", FIRST, PROFILE)
            with pytest.raises(JournalError) as raised:
                await first.accept(SESSION, "hello", FIRST, PROFILE)
            error = raised.value
        assert error.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert comparisons == []
        assert (await journal.load_head(SESSION)).version == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("corrupt", [future_version, other_state])
def test_cas_loser_cross_checks_receipt_against_refreshed_head(
    corrupt: Callable[[SubmissionAcceptance], SubmissionAcceptance],
) -> None:
    async def scenario() -> None:
        journal = CompetingJournal()
        journal.corrupt = corrupt
        journal.synchronize_next_two_commits()

        first, second = await asyncio.gather(
            service(journal, "one").accept(SESSION, "hello", FIRST, PROFILE),
            service(journal, "two").accept(SESSION, "hello", FIRST, PROFILE),
            return_exceptions=True,
        )

        winner, loser = (first, second) if isinstance(first, AcceptedRun) else (second, first)
        assert isinstance(winner, AcceptedRun)
        assert isinstance(loser, JournalError)
        assert loser.failure.code is RunErrorCode.INVARIANT_VIOLATION
        assert journal.head_reads == 3
        snapshot = await journal.load_session(SESSION)
        assert (snapshot.version, snapshot.state, snapshot.sequence) == (1, winner.state, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("starts_terminal", [False, True])
@pytest.mark.parametrize(
    ("second_submission", "second_message"),
    [(FIRST, "hello"), (FIRST, "changed"), (SECOND, "hello")],
)
def test_competing_submissions_resolve_one_atomic_cas(
    starts_terminal: bool, second_submission: SubmissionId, second_message: str
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
            service(journal, "two").accept(SESSION, second_message, second_submission, PROFILE),
            return_exceptions=True,
        )

        snapshot = await journal.load_session(SESSION, after_sequence=before.sequence)
        assert snapshot.version == before.version + 1
        assert snapshot.sequence == before.sequence + 2
        assert len(snapshot.events) == len(snapshot.transcript) == 1
        if second_submission == FIRST and second_message == "hello":
            assert isinstance(first, AcceptedRun)
            assert second == first
            assert snapshot.state == first.state
            assert journal.head_reads == (4 if starts_terminal else 3)
        else:
            winner, loser = (first, second) if isinstance(first, AcceptedRun) else (second, first)
            assert isinstance(winner, AcceptedRun)
            assert isinstance(loser, JournalError)
            assert loser.failure.code is RunErrorCode.JOURNAL_CONFLICT
            assert snapshot.state == winner.state
            if second_submission == FIRST:
                expected = "hello" if winner is first else second_message
                assert snapshot.transcript[0].content == expected
            else:
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
