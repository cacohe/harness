from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps

import pytest
from src.application.submissions import AcceptedRun
from src.core.commands import (
    AdvanceRun,
    CancelRun,
    CommandContext,
    DecideToolCall,
    FinishModelCall,
    ResumeRun,
    RunLimits,
    StartModelCall,
    StartStep,
    StartToolCall,
    ToolCallRequest,
)
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import JournalHead, JournalSnapshot, TranscriptEntry
from src.core.run import RunOutcome
from src.core.state import RunState, RunStatus, TaskStateMachine

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)
LIMITS = RunLimits(8, 60.0, 1024)


def async_test(function):
    @wraps(function)
    def run():
        return asyncio.run(function())

    return run


class Ids:
    def __init__(self) -> None:
        self._value = 0

    def _next(self, prefix: str) -> str:
        self._value += 1
        return f"{prefix}-{self._value}"

    def run(self):
        return RunId(self._next("run"))

    def session(self):
        return SessionId(self._next("session"))

    def submission(self):
        return SubmissionId(self._next("submission"))

    def profile(self):
        return ProfileId(self._next("profile"))

    def event(self):
        return EventId(self._next("event"))

    def step(self):
        return self._next("step")

    def model_call(self):
        return self._next("model")


class Clock:
    def now(self):
        return NOW

    def monotonic(self):
        return 0.0


def state(session: SessionId, submission: SubmissionId, run: RunId | None = None) -> RunState:
    return RunState(
        run or RunId(f"run-{submission.value}"),
        session,
        submission,
        ProfileId("profile"),
        LIMITS,
        version=1,
        status=RunStatus.RUNNING,
        journal_version=1,
        journal_sequence=2,
    )


class Submissions:
    def __init__(self, journal) -> None:
        self.journal = journal
        self.calls = 0

    async def accept(self, session_id, message, submission_id, profile_id):
        self.calls += 1
        current = state(session_id, submission_id)
        self.journal.heads[session_id.value] = current
        self.journal.runs[current.run_id.value] = current
        self.journal.messages[current.run_id.value] = message
        return AcceptedRun(current, 1, True)


class Journal:
    def __init__(self) -> None:
        self.heads: dict[str, RunState] = {}
        self.messages: dict[str, str] = {}
        self.runs: dict[str, RunState] = {}

    async def load_head(self, session_id):
        current = self.heads.get(session_id.value)
        return JournalHead(
            0 if current is None else current.journal_version,
            0 if current is None else current.journal_sequence,
            current,
        )

    async def load_session(self, session_id, after_sequence=0):
        current = self.heads.get(session_id.value)
        transcript = ()
        if current is not None:
            transcript = (
                TranscriptEntry(
                    current.session_id,
                    current.run_id,
                    current.submission_id,
                    "user",
                    self.messages[current.run_id.value],
                    2,
                ),
            )
        return JournalSnapshot(
            session_id,
            0 if current is None else current.journal_version,
            0 if current is None else current.journal_sequence,
            current,
            transcript=transcript,
        )

    async def find_run_input(self, session_id, run_id, submission_id):
        current = self.heads.get(session_id.value)
        if current is None or current.run_id != run_id or current.submission_id != submission_id:
            return None
        return TranscriptEntry(
            current.session_id,
            current.run_id,
            current.submission_id,
            "user",
            self.messages[current.run_id.value],
            2,
        )

    async def find_run_state(self, session_id, run_id, submission_id):
        value = self.runs.get(run_id.value)
        if value is None or value.session_id != session_id or value.submission_id != submission_id:
            return None
        return value


class CompletingCoordinator:
    def __init__(self, journal: Journal, run: RunState, calls: list[str], gate=None) -> None:
        self.journal = journal
        self.run = run
        self.calls = calls
        self.gate = gate

    async def advance(self, state, command):
        self.calls.append(state.session_id.value)
        if self.gate is not None:
            await self.gate.wait()
        completed = replace(
            state,
            status=RunStatus.COMPLETED,
            final_text="done",
            version=state.version + 1,
            journal_version=state.journal_version + 1,
        )
        self.journal.heads[state.session_id.value] = completed
        self.journal.runs[state.run_id.value] = completed
        return RunOutcome(completed, completed.journal_version, completed.journal_sequence)


@async_test
async def test_post_commit_observer_runs_after_acceptance_and_advance_and_is_isolated():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    observed = []

    class Observer:
        async def observe(self, observation):
            head = await journal.load_head(SessionId(observation.session_id))
            assert head.state is not None
            assert head.version == observation.journal_version
            observed.append((observation.kind, observation.status))
            if observation.kind == "submission.accepted":
                raise RuntimeError("observer failure must not affect durable work")

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=2,
        max_waiters_per_session=0,
        observer=Observer(),
    )
    result = await runtime.submit(
        SessionId("observed"), ProfileId("profile"), "message", SubmissionId("submission")
    )
    assert result.status is RunStatus.COMPLETED
    assert observed == [
        ("submission.accepted", "running"),
        ("run.advanced", "completed"),
    ]


@async_test
async def test_replayed_acceptance_emits_no_acceptance_observation_and_hung_observer_times_out():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    observed = []

    class ReplaySubmissions(Submissions):
        async def accept(self, session_id, message, submission_id, profile_id):
            accepted = await super().accept(session_id, message, submission_id, profile_id)
            return AcceptedRun(accepted.state, accepted.version, False)

    class Observer:
        async def observe(self, observation):
            observed.append(observation.kind)
            await asyncio.sleep(10)

    runtime = RuntimeCoordinator(
        submissions=ReplaySubmissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=0,
        observer=Observer(),
        observer_timeout_seconds=0.01,
    )
    result = await asyncio.wait_for(
        runtime.submit(
            SessionId("replay"), ProfileId("profile"), "message", SubmissionId("submission")
        ),
        timeout=0.2,
    )
    assert result.status is RunStatus.COMPLETED
    assert observed == ["run.advanced"]


def waiting_state(session: SessionId, submission: SubmissionId) -> RunState:
    current = state(session, submission)
    context = CommandContext(NOW, 0, (EventId("event"),))
    commands = (
        StartStep(current.run_id, session, "step", context=context),
        StartModelCall(current.run_id, session, "step", "model", context=context),
        FinishModelCall(
            current.run_id,
            session,
            "step",
            "model",
            "",
            (
                ToolCallRequest(
                    "tool-call",
                    "write",
                    "{}",
                    requires_approval=True,
                    approval_scope="workspace-write",
                    workspace_id="workspace",
                ),
            ),
            context=context,
        ),
        StartToolCall(
            current.run_id,
            session,
            "step",
            "model",
            "tool-call",
            True,
            context=context,
        ),
    )
    for command in commands:
        current = TaskStateMachine.apply(current, command).state
    return replace(current, journal_version=2, journal_sequence=3)


@async_test
async def test_same_session_is_fifo_but_different_sessions_run_concurrently():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    ids = Ids()
    entered: list[str] = []
    releases = {"a": asyncio.Event(), "b": asyncio.Event()}

    def factory(run, _message):
        return CompletingCoordinator(journal, run, entered, releases[run.session_id.value])

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=factory,
        ids=ids,
        clock=Clock(),
        max_sessions=4,
        max_waiters_per_session=2,
    )
    profile = ProfileId("profile")
    first = asyncio.create_task(runtime.submit(SessionId("a"), profile, "one", SubmissionId("one")))
    await asyncio.sleep(0)
    queued = asyncio.create_task(
        runtime.submit(SessionId("a"), profile, "two", SubmissionId("two"))
    )
    other = asyncio.create_task(
        runtime.submit(SessionId("b"), profile, "other", SubmissionId("other"))
    )
    await asyncio.sleep(0)
    assert entered == ["a", "b"]
    releases["b"].set()
    await other
    releases["a"].set()
    await first
    await queued
    assert entered == ["a", "b", "a"]


@async_test
async def test_waiter_cancellation_releases_fifo_slot_and_capacity_is_bounded():
    from src.application.runtime import RuntimeBusyError, RuntimeCoordinator

    journal = Journal()
    blocker = asyncio.Event()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, [], blocker),
        ids=Ids(),
        clock=Clock(),
        max_sessions=2,
        max_waiters_per_session=1,
    )
    args = (SessionId("a"), ProfileId("profile"), "x")
    active = asyncio.create_task(runtime.submit(*args, SubmissionId("one")))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(runtime.submit(*args, SubmissionId("two")))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeBusyError):
        await runtime.submit(*args, SubmissionId("three"))
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    replacement = asyncio.create_task(runtime.submit(*args, SubmissionId("four")))
    blocker.set()
    await active
    await replacement


@async_test
async def test_close_cancels_active_rejects_waiters_and_is_idempotent():
    from src.application.runtime import ApplicationClosedError, RuntimeCoordinator

    journal = Journal()
    blocker = asyncio.Event()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, [], blocker),
        ids=Ids(),
        clock=Clock(),
        max_sessions=2,
        max_waiters_per_session=2,
    )
    args = (SessionId("a"), ProfileId("profile"), "x")
    active = asyncio.create_task(runtime.submit(*args, SubmissionId("one")))
    waiter = asyncio.create_task(runtime.submit(*args, SubmissionId("two")))
    await asyncio.sleep(0)
    await runtime.close()
    with pytest.raises(asyncio.CancelledError):
        await active
    with pytest.raises(ApplicationClosedError):
        await waiter
    await runtime.close()
    with pytest.raises(ApplicationClosedError):
        await runtime.submit(*args, SubmissionId("three"))


@async_test
async def test_owned_resources_close_reverse_once_but_injected_are_untouched():
    from src.application.runtime import RuntimeCoordinator

    order: list[str] = []

    class Resource:
        def __init__(self, name):
            self.name = name

        async def aclose(self):
            order.append(self.name)

    owned = (Resource("first"), Resource("second"))
    injected = Resource("injected")
    journal = Journal()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=0,
        owned_resources=owned,
    )
    runtime.keep_alive(injected)
    await runtime.close()
    await runtime.close()
    assert order == ["second", "first"]


@async_test
async def test_cancelling_close_caller_does_not_cancel_exactly_once_cleanup():
    from src.application.runtime import RuntimeCoordinator

    entered = asyncio.Event()
    release = asyncio.Event()
    closes = 0

    class Resource:
        async def aclose(self):
            nonlocal closes
            closes += 1
            entered.set()
            await release.wait()

    journal = Journal()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=0,
        owned_resources=(Resource(),),
    )
    first = asyncio.create_task(runtime.close())
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(runtime.close())
    release.set()
    await second
    assert closes == 1


@async_test
async def test_active_exception_releases_session_for_the_next_fifo_operation():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    attempts = 0

    class RaisesOnce:
        async def advance(self, state, _command):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("boom")
            completed = replace(
                state,
                status=RunStatus.COMPLETED,
                final_text="done",
                version=state.version + 1,
                journal_version=state.journal_version + 1,
            )
            journal.heads[state.session_id.value] = completed
            return RunOutcome(completed, completed.journal_version, completed.journal_sequence)

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda _run, _message: RaisesOnce(),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    with pytest.raises(RuntimeError, match="boom"):
        await runtime.submit(SessionId("a"), ProfileId("profile"), "one", SubmissionId("one"))
    result = await runtime.submit(SessionId("a"), ProfileId("profile"), "two", SubmissionId("two"))
    assert result.status is RunStatus.COMPLETED


@async_test
async def test_approval_rejects_cross_session_run_and_tool_ownership_before_io():
    from src.application.runtime import RuntimeCoordinator, SessionOwnershipError

    journal = Journal()
    current = waiting_state(SessionId("a"), SubmissionId("submission"))
    journal.heads["a"] = current
    journal.messages[current.run_id.value] = "hello"
    factory_calls = 0

    def factory(_run, _message):
        nonlocal factory_calls
        factory_calls += 1
        return CompletingCoordinator(journal, current, [])

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=factory,
        ids=Ids(),
        clock=Clock(),
        max_sessions=2,
        max_waiters_per_session=1,
    )
    with pytest.raises(SessionOwnershipError):
        await runtime.decide_tool_call(SessionId("a"), RunId("other"), "tool-call", False)
    with pytest.raises(SessionOwnershipError):
        await runtime.decide_tool_call(SessionId("a"), current.run_id, "other-call", False)
    with pytest.raises(SessionOwnershipError):
        await runtime.decide_tool_call(SessionId("b"), current.run_id, "tool-call", False)
    assert factory_calls == 0


@async_test
async def test_approval_decides_resumes_and_drives_one_step_at_a_time():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    current = waiting_state(SessionId("a"), SubmissionId("submission"))
    journal.heads["a"] = current
    journal.messages[current.run_id.value] = "hello"
    commands: list[type[object]] = []

    class ApprovalCoordinator:
        async def advance(self, run, command):
            commands.append(type(command))
            step = run.current_step
            assert step is not None
            tools = step.tool_calls
            if isinstance(command, DecideToolCall):
                tool = replace(tools[0], approval=replace(tools[0].approval, approved=True))
                next_state = replace(run, current_step=replace(step, tool_calls=(tool,)))
            elif isinstance(command, ResumeRun):
                tool = replace(tools[0], approval=replace(tools[0].approval, resumed=True))
                next_state = replace(
                    run,
                    status=RunStatus.RUNNING,
                    current_step=replace(step, tool_calls=(tool,)),
                )
            else:
                assert isinstance(command, AdvanceRun)
                next_state = replace(
                    run,
                    status=RunStatus.COMPLETED,
                    current_step=None,
                    final_text="done",
                )
            next_state = replace(
                next_state,
                version=run.version + 1,
                journal_version=run.journal_version + 1,
            )
            journal.heads["a"] = next_state
            return RunOutcome(next_state, next_state.journal_version, next_state.journal_sequence)

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda _run, _message: ApprovalCoordinator(),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    result = await runtime.decide_tool_call(SessionId("a"), current.run_id, "tool-call", True)
    assert result.status is RunStatus.COMPLETED
    assert commands == [DecideToolCall, ResumeRun, AdvanceRun]


@async_test
async def test_durable_approval_decision_resumes_after_observer_cancellation_and_restart():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    current = waiting_state(SessionId("a"), SubmissionId("submission"))
    journal.heads["a"] = current
    journal.runs[current.run_id.value] = current
    journal.messages[current.run_id.value] = "hello"
    commands: list[type[object]] = []

    class ApprovalCoordinator:
        async def advance(self, run, command):
            commands.append(type(command))
            step = run.current_step
            assert step is not None
            tool = step.tool_calls[0]
            if isinstance(command, DecideToolCall):
                tool = replace(tool, approval=replace(tool.approval, approved=True))
                next_state = replace(run, current_step=replace(step, tool_calls=(tool,)))
            elif isinstance(command, ResumeRun):
                tool = replace(tool, approval=replace(tool.approval, resumed=True))
                next_state = replace(
                    run,
                    status=RunStatus.RUNNING,
                    current_step=replace(step, tool_calls=(tool,)),
                )
            else:
                assert isinstance(command, AdvanceRun)
                next_state = replace(
                    run, status=RunStatus.COMPLETED, current_step=None, final_text="done"
                )
            next_state = replace(
                next_state,
                version=run.version + 1,
                journal_version=run.journal_version + 1,
            )
            journal.heads["a"] = next_state
            journal.runs[run.run_id.value] = next_state
            return RunOutcome(next_state, next_state.journal_version, next_state.journal_sequence)

    class CancelObserver:
        async def observe(self, _observation):
            raise asyncio.CancelledError

    first = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda *_args: ApprovalCoordinator(),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
        observer=CancelObserver(),
    )
    with pytest.raises(asyncio.CancelledError):
        await first.decide_tool_call(SessionId("a"), current.run_id, "tool-call", True)
    assert journal.heads["a"].status is RunStatus.WAITING_APPROVAL

    class Replay:
        async def accept(self, *_args):
            return AcceptedRun(state(SessionId("a"), SubmissionId("submission")), 1, False)

    restarted = RuntimeCoordinator(
        submissions=Replay(),
        journal=journal,
        coordinator_factory=lambda *_args: ApprovalCoordinator(),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    result = await restarted.submit(
        SessionId("a"), ProfileId("profile"), "hello", SubmissionId("submission")
    )
    assert result.status is RunStatus.COMPLETED
    assert commands == [DecideToolCall, ResumeRun, AdvanceRun]


@async_test
async def test_duplicate_terminal_submission_does_not_drive_or_append_again():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    coordinator_calls: list[str] = []

    class DuplicateSubmissions(Submissions):
        accepted: AcceptedRun | None = None

        async def accept(self, session_id, message, submission_id, profile_id):
            if self.accepted is None:
                self.accepted = await super().accept(session_id, message, submission_id, profile_id)
            return self.accepted

    runtime = RuntimeCoordinator(
        submissions=DuplicateSubmissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(
            journal, run, coordinator_calls
        ),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    args = (SessionId("a"), ProfileId("profile"), "hello", SubmissionId("same"))
    first = await runtime.submit(*args)
    second = await runtime.submit(*args)
    assert first.status is second.status is RunStatus.COMPLETED
    assert coordinator_calls == ["a"]


@async_test
async def test_cancel_interrupts_blocked_drive_and_commits_once():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    entered = asyncio.Event()
    interrupted = asyncio.Event()
    cancel_entered = asyncio.Event()
    allow_cancel_commit = asyncio.Event()
    commands: list[type[object]] = []

    class InterruptibleCoordinator:
        async def advance(self, run, command):
            commands.append(type(command))
            if isinstance(command, AdvanceRun):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    interrupted.set()
                    raise
            assert isinstance(command, CancelRun)
            cancel_entered.set()
            await allow_cancel_commit.wait()
            cancelled = replace(
                run,
                status=RunStatus.CANCELLED,
                failure=RunFailure(RunErrorCode.CANCELLED),
                version=run.version + 1,
                journal_version=run.journal_version + 1,
            )
            journal.heads[run.session_id.value] = cancelled
            journal.runs[run.run_id.value] = cancelled
            return RunOutcome(cancelled, cancelled.journal_version, cancelled.journal_sequence)

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda _run, _message: InterruptibleCoordinator(),
        ids=Ids(),
        clock=Clock(),
        max_sessions=2,
        max_waiters_per_session=1,
    )
    submit = asyncio.create_task(
        runtime.submit(SessionId("a"), ProfileId("profile"), "hello", SubmissionId("one"))
    )
    await entered.wait()
    run_id = journal.heads["a"].run_id
    from src.application.runtime import SessionOwnershipError

    with pytest.raises(SessionOwnershipError):
        await runtime.cancel_run(SessionId("a"), RunId("wrong"))
    with pytest.raises(SessionOwnershipError):
        await runtime.cancel_run(SessionId("other"), run_id)
    assert not interrupted.is_set()
    caller = asyncio.create_task(runtime.cancel_run(SessionId("a"), run_id))
    await cancel_entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    allow_cancel_commit.set()
    cancelled = await asyncio.wait_for(runtime.cancel_run(SessionId("a"), run_id), 1)
    with pytest.raises(asyncio.CancelledError):
        await submit
    assert interrupted.is_set()
    assert cancelled.status is RunStatus.CANCELLED
    assert commands.count(CancelRun) == 1


@async_test
async def test_cancel_completion_race_returns_the_existing_terminal_outcome():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    completed = replace(
        state(SessionId("a"), SubmissionId("one")),
        status=RunStatus.COMPLETED,
        final_text="won race",
    )
    journal.heads["a"] = completed
    journal.runs[completed.run_id.value] = completed
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda *_args: pytest.fail("terminal cancellation must not drive"),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    assert await runtime.cancel_run(SessionId("a"), completed.run_id) == completed


@async_test
async def test_historical_submission_replay_returns_its_terminal_run_without_driving():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    original = state(SessionId("a"), SubmissionId("first"), RunId("original"))
    original_done = replace(
        original,
        status=RunStatus.COMPLETED,
        final_text="original result",
        version=2,
        journal_version=2,
    )
    newer = replace(
        state(SessionId("a"), SubmissionId("second"), RunId("newer")),
        status=RunStatus.COMPLETED,
        final_text="newer result",
        journal_version=4,
    )
    journal.heads["a"] = newer
    journal.runs[original.run_id.value] = original_done
    journal.runs[newer.run_id.value] = newer
    journal.messages[original.run_id.value] = "old"

    class Replay:
        async def accept(self, *_args):
            return AcceptedRun(original, 1, False)

    runtime = RuntimeCoordinator(
        submissions=Replay(),
        journal=journal,
        coordinator_factory=lambda *_args: pytest.fail("historical replay must not drive"),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    result = await runtime.submit(
        SessionId("a"), ProfileId("profile"), "old", SubmissionId("first")
    )
    assert result == original_done
    assert journal.heads["a"] == newer


@async_test
async def test_session_registry_is_bounded_and_explicit_close_reclaims_capacity():
    from src.application.runtime import RuntimeCoordinator, SessionCapacityError

    journal = Journal()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=0,
    )
    await runtime.open_session(SessionId("one"))
    with pytest.raises(SessionCapacityError):
        await runtime.open_session(SessionId("two"))
    await runtime.close_session(SessionId("one"))
    await runtime.open_session(SessionId("two"))


@async_test
async def test_public_id_boundaries_reject_subclasses_and_corrupt_exact_ids_without_hooks():
    from src.application.runtime import RuntimeCoordinator, SessionOwnershipError

    class HostileSessionId(SessionId):
        @property
        def value(self):
            raise AssertionError("hostile value descriptor executed")

    hostile_session = object.__new__(HostileSessionId)

    class HostileText(str):
        def __eq__(self, _other):
            raise AssertionError("hostile equality executed")

        def __hash__(self):
            raise AssertionError("hostile hash executed")

    corrupt_run = RunId("run")
    object.__setattr__(corrupt_run, "value", HostileText("run"))
    corrupt_profile = ProfileId("profile")
    object.__setattr__(corrupt_profile, "value", object())
    corrupt_submission = SubmissionId("submission")
    object.__setattr__(corrupt_submission, "value", object())
    journal = Journal()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=0,
    )

    for operation in (
        runtime.open_session(hostile_session),  # type: ignore[arg-type]
        runtime.close_session(hostile_session),  # type: ignore[arg-type]
        runtime.submit(
            SessionId("session"),
            corrupt_profile,
            "message",
            corrupt_submission,
        ),
        runtime.decide_tool_call(SessionId("session"), corrupt_run, "call", True),
    ):
        with pytest.raises((SessionOwnershipError, ValueError)):
            await operation


@async_test
async def test_close_and_reopen_invalidates_owner_before_operation_registration():
    from src.application.runtime import ApplicationClosedError, RuntimeCoordinator

    journal = Journal()
    submissions = Submissions(journal)
    runtime = RuntimeCoordinator(
        submissions=submissions,
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    reached = asyncio.Event()
    release = asyncio.Event()
    original = runtime._operation

    @asynccontextmanager
    async def blocked_registration(*args):
        reached.set()
        await release.wait()
        async with original(*args):
            yield

    runtime._operation = blocked_registration  # type: ignore[method-assign]
    old = asyncio.create_task(
        runtime.submit(SessionId("same"), ProfileId("profile"), "old", SubmissionId("old"))
    )
    await reached.wait()
    await runtime.close_session(SessionId("same"))
    await runtime.open_session(SessionId("same"))
    release.set()
    with pytest.raises(ApplicationClosedError):
        await old
    assert submissions.calls == 0


@async_test
async def test_cancelled_pre_registration_owner_releases_gate_without_deadlock():
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    reached = asyncio.Event()
    release = asyncio.Event()
    original = runtime._operation

    @asynccontextmanager
    async def blocked_registration(*args):
        reached.set()
        await release.wait()
        async with original(*args):
            yield

    runtime._operation = blocked_registration  # type: ignore[method-assign]
    cancelled = asyncio.create_task(
        runtime.submit(SessionId("same"), ProfileId("profile"), "old", SubmissionId("old"))
    )
    await reached.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    runtime._operation = original  # type: ignore[method-assign]
    result = await asyncio.wait_for(
        runtime.submit(SessionId("same"), ProfileId("profile"), "new", SubmissionId("new")),
        timeout=1,
    )
    assert result.status is RunStatus.COMPLETED


@async_test
async def test_close_session_cancels_active_rejects_queue_and_allows_clean_reopen():
    from src.application.runtime import ApplicationClosedError, RuntimeCoordinator

    journal = Journal()
    blocker = asyncio.Event()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, [], blocker),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    active = asyncio.create_task(
        runtime.submit(SessionId("same"), ProfileId("profile"), "active", SubmissionId("active"))
    )
    await asyncio.sleep(0)
    queued = asyncio.create_task(
        runtime.submit(SessionId("same"), ProfileId("profile"), "queued", SubmissionId("queued"))
    )
    await asyncio.sleep(0)

    await runtime.close_session(SessionId("same"))

    with pytest.raises(asyncio.CancelledError):
        await active
    with pytest.raises(ApplicationClosedError):
        await queued
    blocker.set()
    reopened = await asyncio.wait_for(
        runtime.submit(SessionId("same"), ProfileId("profile"), "new", SubmissionId("new")),
        timeout=1,
    )
    assert reopened.status is RunStatus.COMPLETED


@async_test
async def test_approval_uses_bounded_run_input_lookup_not_full_session_history():
    from src.application.runtime import RuntimeCoordinator

    class BoundedJournal(Journal):
        async def load_session(self, *_args, **_kwargs):
            raise AssertionError("approval must not materialize Session history")

    journal = BoundedJournal()
    current = waiting_state(SessionId("a"), SubmissionId("submission"))
    journal.heads["a"] = current
    journal.messages[current.run_id.value] = "hello"

    class StopsAfterDecision:
        async def advance(self, run, command):
            if isinstance(command, DecideToolCall):
                step = run.current_step
                assert step is not None
                tool = replace(
                    step.tool_calls[0],
                    approval=replace(step.tool_calls[0].approval, approved=False),
                )
                next_state = replace(run, current_step=replace(step, tool_calls=(tool,)))
            else:
                next_state = replace(
                    run, status=RunStatus.COMPLETED, current_step=None, final_text="denied"
                )
            next_state = replace(
                next_state,
                version=run.version + 1,
                journal_version=run.journal_version + 1,
            )
            journal.heads["a"] = next_state
            return RunOutcome(next_state, next_state.journal_version, next_state.journal_sequence)

    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda _run, _message: StopsAfterDecision(),
        ids=Ids(),
        clock=Clock(),
        max_sessions=1,
        max_waiters_per_session=1,
    )
    result = await runtime.decide_tool_call(SessionId("a"), current.run_id, "tool-call", False)
    assert result.status is RunStatus.COMPLETED
