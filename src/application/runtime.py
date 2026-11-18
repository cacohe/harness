from __future__ import annotations

import asyncio
import inspect
import math
from collections import deque
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, TypeVar

from src.application.observability import Observation, PostCommitObserver
from src.application.submissions import AcceptedRun, SubmissionService
from src.core.commands import AdvanceRun, CancelRun, CommandContext, DecideToolCall, ResumeRun
from src.core.ids import Clock, IdGenerator, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import JournalHead, RunJournal
from src.core.receipt_snapshot import snapshot_run_state, snapshot_transcript
from src.core.run import RunCoordinator, RunOutcome
from src.core.state import RunState, RunStatus


class ApplicationClosedError(RuntimeError):
    pass


class RuntimeBusyError(RuntimeError):
    pass


class SessionCapacityError(RuntimeError):
    pass


class SessionOwnershipError(ValueError):
    pass


class CoordinatorFactory(Protocol):
    def __call__(
        self, state: RunState, user_input: str
    ) -> RunCoordinator | Awaitable[RunCoordinator]: ...


class _Gate:
    def __init__(self, max_waiters: int) -> None:
        self._max_waiters = max_waiters
        self._lock = asyncio.Lock()
        self._active = False
        self._closed = False
        self._waiters: deque[asyncio.Future[None]] = deque()

    @asynccontextmanager
    async def enter(self):
        waiter: asyncio.Future[None] | None = None
        async with self._lock:
            if self._closed:
                raise ApplicationClosedError("Session is closed")
            if not self._active and not self._waiters:
                self._active = True
            else:
                if len(self._waiters) >= self._max_waiters:
                    raise RuntimeBusyError("Session operation queue is full")
                waiter = asyncio.get_running_loop().create_future()
                self._waiters.append(waiter)
        if waiter is not None:
            try:
                await waiter
            except BaseException:
                await self._abandon(waiter)
                raise
        try:
            yield
        finally:
            await self._release()

    async def _abandon(self, waiter: asyncio.Future[None]) -> None:
        async with self._lock:
            try:
                self._waiters.remove(waiter)
                return
            except ValueError:
                pass
            # The waiter was already granted ownership. Hand it to the next waiter.
            if waiter.done() and not waiter.cancelled() and waiter.exception() is None:
                self._grant_next()

    async def _release(self) -> None:
        async with self._lock:
            self._grant_next()

    def _grant_next(self) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            waiter.set_result(None)
            return
        self._active = False

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            while self._waiters:
                waiter = self._waiters.popleft()
                if not waiter.done():
                    waiter.set_exception(ApplicationClosedError("Session is closed"))


@dataclass(slots=True, eq=False)
class _SessionSlot:
    gate: _Gate


class RuntimeCoordinator:
    """Own Session serialization and drive the Core one durable step at a time."""

    def __init__(
        self,
        *,
        submissions: SubmissionService,
        journal: RunJournal,
        coordinator_factory: CoordinatorFactory,
        ids: IdGenerator,
        clock: Clock,
        max_sessions: int,
        max_waiters_per_session: int,
        max_advances_per_operation: int = 1024,
        owned_resources: tuple[object, ...] = (),
        observer: PostCommitObserver | None = None,
        observer_timeout_seconds: float = 0.1,
    ) -> None:
        if (
            type(max_sessions) is not int
            or not 1 <= max_sessions <= 1_000_000
            or type(max_waiters_per_session) is not int
            or not 0 <= max_waiters_per_session <= 1_000_000
            or type(max_advances_per_operation) is not int
            or not 1 <= max_advances_per_operation <= 1_000_000
            or type(owned_resources) is not tuple
            or len({id(item) for item in owned_resources}) != len(owned_resources)
            or isinstance(observer_timeout_seconds, bool)
            or type(observer_timeout_seconds) not in (int, float)
            or not math.isfinite(observer_timeout_seconds)
            or not 0.001 <= observer_timeout_seconds <= 30
        ):
            raise ValueError("Runtime limits and resources are invalid")
        self._submissions = submissions
        self._journal = journal
        self._coordinator_factory = coordinator_factory
        self._ids = ids
        self._clock = clock
        self._max_sessions = max_sessions
        self._max_waiters = max_waiters_per_session
        self._max_advances = max_advances_per_operation
        self._owned_resources = owned_resources
        self._observer = observer
        self._observer_timeout = float(observer_timeout_seconds)
        self._sessions: dict[str, _SessionSlot] = {}
        self._registry_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._active: dict[asyncio.Task[object], _SessionSlot] = {}
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def keep_alive(self, _resource: object) -> None:
        """Document a caller-owned dependency without taking close ownership."""
        if self._closing or self._closed:
            raise ApplicationClosedError("Runtime is closed")

    async def open_session(self, session_id: SessionId) -> None:
        session_id = _canonical_id(session_id, SessionId, "Session identity is invalid")
        await self._slot(session_id)

    async def close_session(self, session_id: SessionId) -> None:
        session_id = _canonical_id(session_id, SessionId, "Session identity is invalid")
        key = _id_value(session_id)
        async with self._registry_lock:
            slot = self._sessions.pop(key, None)
            if slot is not None:
                async with self._lifecycle_lock:
                    current = asyncio.current_task()
                    tasks = tuple(
                        task
                        for task, active_slot in self._active.items()
                        if active_slot is slot and task is not current
                    )
        if slot is None:
            return
        await slot.gate.close()
        await _cancel_and_wait(tasks)

    async def submit(
        self,
        session_id: SessionId,
        profile_id: ProfileId,
        message: str,
        submission_id: SubmissionId,
    ) -> RunState:
        session_id, profile_id, submission_id = _validate_submit(
            session_id, profile_id, message, submission_id
        )
        slot = await self._slot(session_id)
        async with slot.gate.enter():
            async with self._operation(session_id, slot):
                accepted = await self._submissions.accept(
                    session_id, message, submission_id, profile_id
                )
                state = await self._accepted_head(accepted)
                if accepted.committed:
                    await self._observe("submission.accepted", state)
                return await self._drive(state, message)

    async def decide_tool_call(
        self,
        session_id: SessionId,
        run_id: RunId,
        tool_call_id: str,
        approved: bool,
    ) -> RunState:
        session_id = _canonical_id(session_id, SessionId, "Approval identity is invalid")
        run_id = _canonical_id(run_id, RunId, "Approval identity is invalid")
        if type(tool_call_id) is not str or not tool_call_id or type(approved) is not bool:
            raise SessionOwnershipError("Approval identity is invalid")
        slot = await self._slot(session_id)
        async with slot.gate.enter():
            async with self._operation(session_id, slot):
                state = await self._head_state(session_id)
                call_id = _pending_call(state, session_id, run_id)
                if call_id != tool_call_id:
                    raise SessionOwnershipError("Tool call is not owned by this Session and Run")
                message = await self._user_input(state)
                coordinator = await self._coordinator(state, message)
                outcome = await coordinator.advance(
                    state,
                    DecideToolCall(
                        state.run_id,
                        state.session_id,
                        tool_call_id,
                        approved,
                        context=self._context(state),
                    ),
                )
                decided = _outcome_state(outcome, state)
                await self._observe("run.advanced", decided)
                resumed_coordinator = await self._coordinator(decided, message)
                resumed = await resumed_coordinator.advance(
                    decided,
                    ResumeRun(
                        decided.run_id,
                        decided.session_id,
                        context=self._context(decided),
                    ),
                )
                resumed_state = _outcome_state(resumed, decided)
                await self._observe("run.advanced", resumed_state)
                return await self._drive(resumed_state, message)

    async def cancel_run(self, session_id: SessionId, run_id: RunId) -> RunState:
        session_id = _canonical_id(session_id, SessionId, "Cancellation identity is invalid")
        run_id = _canonical_id(run_id, RunId, "Cancellation identity is invalid")
        slot = await self._slot(session_id)
        async with slot.gate.enter():
            async with self._operation(session_id, slot):
                state = await self._head_state(session_id)
                if state.session_id != session_id or state.run_id != run_id:
                    raise SessionOwnershipError("Run is not owned by this Session")
                if state.status.terminal:
                    return state
                message = await self._user_input(state)
                coordinator = await self._coordinator(state, message)
                outcome = await coordinator.advance(
                    state,
                    CancelRun(
                        state.run_id,
                        state.session_id,
                        context=self._context(state),
                    ),
                )
                cancelled = _outcome_state(outcome, state)
                await self._observe("run.advanced", cancelled)
                return cancelled

    async def close(self) -> None:
        async with self._lifecycle_lock:
            task = self._close_task
            if task is None:
                self._closing = True
                task = asyncio.create_task(self._close_impl())
                self._close_task = task
        await asyncio.shield(task)

    async def _close_impl(self) -> None:
        failure: BaseException | None = None
        try:
            async with self._registry_lock:
                slots = tuple(self._sessions.values())
                self._sessions.clear()
            await asyncio.gather(*(slot.gate.close() for slot in slots))
            current = asyncio.current_task()
            async with self._lifecycle_lock:
                tasks = tuple(task for task in self._active if task is not current)
            await _cancel_and_wait(tasks)
            for resource in reversed(self._owned_resources):
                try:
                    await _close_resource(resource)
                except BaseException as error:
                    if failure is None:
                        failure = error
        finally:
            async with self._lifecycle_lock:
                self._closed = True
        if failure is not None:
            raise failure

    async def _slot(self, session_id: SessionId) -> _SessionSlot:
        key = _id_value(session_id)
        async with self._registry_lock:
            if self._closing or self._closed:
                raise ApplicationClosedError("Runtime is closed")
            slot = self._sessions.get(key)
            if slot is None:
                if len(self._sessions) >= self._max_sessions:
                    raise SessionCapacityError("Runtime Session capacity is exhausted")
                slot = _SessionSlot(_Gate(self._max_waiters))
                self._sessions[key] = slot
            return slot

    @asynccontextmanager
    async def _operation(self, session_id: SessionId, slot: _SessionSlot):
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Application operation requires an asyncio Task")
        key = _id_value(session_id)
        async with self._registry_lock:
            async with self._lifecycle_lock:
                if self._closing or self._closed or self._sessions.get(key) is not slot:
                    raise ApplicationClosedError("Session is closed")
                self._active[task] = slot
        try:
            yield
        finally:
            async with self._lifecycle_lock:
                self._active.pop(task, None)

    async def _accepted_head(self, accepted: AcceptedRun) -> RunState:
        if type(accepted) is not AcceptedRun:
            raise SessionOwnershipError("Submission acceptance is invalid")
        initial = snapshot_run_state(object.__getattribute__(accepted, "state"))
        head = await self._journal.load_head(initial.session_id)
        state = _head(head, initial.session_id)
        if state.run_id != initial.run_id or state.submission_id != initial.submission_id:
            raise SessionOwnershipError("Submission is no longer the active Run")
        return state

    async def _head_state(self, session_id: SessionId) -> RunState:
        state = _head(await self._journal.load_head(session_id), session_id)
        return state

    async def _user_input(self, state: RunState) -> str:
        value = await self._journal.find_run_input(
            state.session_id, state.run_id, state.submission_id
        )
        if value is None:
            raise SessionOwnershipError("Run does not have one durable user input")
        try:
            entry = snapshot_transcript(value)
        except (AttributeError, TypeError, ValueError):
            raise SessionOwnershipError("Run input is invalid") from None
        if (
            entry.session_id != state.session_id
            or entry.run_id != state.run_id
            or entry.submission_id != state.submission_id
            or entry.role != "user"
            or entry.sequence < 1
        ):
            raise SessionOwnershipError("Run input is not owned by the durable Run")
        return entry.content

    async def _drive(self, state: RunState, message: str) -> RunState:
        for _ in range(self._max_advances):
            if state.status.terminal or state.status is RunStatus.WAITING_APPROVAL:
                return state
            coordinator = await self._coordinator(state, message)
            outcome = await coordinator.advance(
                state,
                AdvanceRun(
                    state.run_id,
                    state.session_id,
                    context=self._context(state),
                ),
            )
            next_state = _outcome_state(outcome, state)
            await self._observe("run.advanced", next_state)
            if next_state.version <= state.version:
                raise RuntimeError("Run coordinator did not advance the state")
            state = next_state
        raise RuntimeError("Application operation exceeded its advance limit")

    async def _coordinator(self, state: RunState, message: str) -> RunCoordinator:
        value = self._coordinator_factory(state, message)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _observe(self, kind: str, state: RunState) -> None:
        observer = self._observer
        if observer is None:
            return
        observation = Observation(
            kind,
            state.session_id.value,
            state.run_id.value,
            state.status.value,
            state.journal_version,
            state.journal_sequence,
        )
        try:
            await asyncio.wait_for(observer.observe(observation), timeout=self._observer_timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _context(self, state: RunState) -> CommandContext:
        return CommandContext(self._clock.now(), state.elapsed_seconds, (self._ids.event(),))


_IdT = TypeVar("_IdT", SessionId, RunId, SubmissionId, ProfileId)


def _canonical_id(value: object, expected: type[_IdT], message: str) -> _IdT:
    try:
        if type(value) is not expected:
            raise ValueError
        raw = object.__getattribute__(value, "value")
        if type(raw) is not str:
            raise ValueError
        return expected(raw)
    except (AttributeError, TypeError, ValueError):
        raise SessionOwnershipError(message) from None


def _id_value(value: SessionId | RunId | SubmissionId | ProfileId) -> str:
    raw = object.__getattribute__(value, "value")
    assert type(raw) is str
    return raw


def _validate_submit(
    session_id: SessionId,
    profile_id: ProfileId,
    message: str,
    submission_id: SubmissionId,
) -> tuple[SessionId, ProfileId, SubmissionId]:
    if type(message) is not str:
        raise ValueError("Submission arguments are invalid")
    return (
        _canonical_id(session_id, SessionId, "Submission arguments are invalid"),
        _canonical_id(profile_id, ProfileId, "Submission arguments are invalid"),
        _canonical_id(submission_id, SubmissionId, "Submission arguments are invalid"),
    )


async def _cancel_and_wait(tasks: tuple[asyncio.Task[object], ...]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _head(value: object, session_id: SessionId) -> RunState:
    try:
        if type(value) is not JournalHead:
            raise ValueError
        version = object.__getattribute__(value, "version")
        sequence = object.__getattribute__(value, "sequence")
        state = snapshot_run_state(object.__getattribute__(value, "state"))
        if (
            type(version) is not int
            or type(sequence) is not int
            or state.session_id != session_id
            or state.journal_version != version
            or state.journal_sequence != sequence
        ):
            raise ValueError
        return state
    except (AttributeError, TypeError, ValueError):
        raise SessionOwnershipError("Journal head is invalid") from None


def _pending_call(state: RunState, session_id: SessionId, run_id: RunId) -> str:
    if (
        state.session_id != session_id
        or state.run_id != run_id
        or state.status is not RunStatus.WAITING_APPROVAL
        or state.current_step is None
        or state.current_step.next_tool_index >= len(state.current_step.tool_calls)
    ):
        raise SessionOwnershipError("Run is not waiting for approval")
    tool = state.current_step.tool_calls[state.current_step.next_tool_index]
    if tool.approval is None or tool.approval.approved is not None:
        raise SessionOwnershipError("Tool call is not awaiting a decision")
    return tool.tool_call_id


def _outcome_state(value: object, previous: RunState) -> RunState:
    try:
        if type(value) is not RunOutcome:
            raise ValueError
        state = snapshot_run_state(object.__getattribute__(value, "state"))
        version = object.__getattribute__(value, "journal_version")
        sequence = object.__getattribute__(value, "sequence")
        if (
            state.run_id != previous.run_id
            or state.session_id != previous.session_id
            or type(version) is not int
            or type(sequence) is not int
            or state.journal_version != version
            or state.journal_sequence != sequence
        ):
            raise ValueError
        return state
    except (AttributeError, TypeError, ValueError):
        raise SessionOwnershipError("Run outcome is invalid") from None


async def _close_resource(resource: object) -> None:
    closer = getattr(resource, "aclose", None)
    if closer is None:
        closer = getattr(resource, "close", None)
    if closer is None:
        return
    result = closer()
    if inspect.isawaitable(result):
        await result
