from __future__ import annotations

import asyncio
import inspect
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from src.application.submissions import AcceptedRun, SubmissionService
from src.core.commands import AdvanceRun, CommandContext, DecideToolCall, ResumeRun
from src.core.ids import Clock, IdGenerator, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import JournalHead, RunJournal
from src.core.receipt_snapshot import snapshot_journal_snapshot, snapshot_run_state
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
    def __call__(self, state: RunState, user_input: str) -> RunCoordinator: ...


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


@dataclass(slots=True)
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
        self._sessions: dict[str, _SessionSlot] = {}
        self._registry_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._active: dict[asyncio.Task[object], str] = {}
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def keep_alive(self, _resource: object) -> None:
        """Document a caller-owned dependency without taking close ownership."""
        if self._closing or self._closed:
            raise ApplicationClosedError("Runtime is closed")

    async def open_session(self, session_id: SessionId) -> None:
        await self._slot(session_id)

    async def close_session(self, session_id: SessionId) -> None:
        key = _session_key(session_id)
        async with self._registry_lock:
            slot = self._sessions.pop(key, None)
        if slot is None:
            return
        await slot.gate.close()
        await self._cancel_tasks(key)

    async def submit(
        self,
        session_id: SessionId,
        profile_id: ProfileId,
        message: str,
        submission_id: SubmissionId,
    ) -> RunState:
        _validate_submit(session_id, profile_id, message, submission_id)
        slot = await self._slot(session_id)
        async with slot.gate.enter():
            async with self._operation(session_id):
                accepted = await self._submissions.accept(
                    session_id, message, submission_id, profile_id
                )
                state = await self._accepted_head(accepted)
                return await self._drive(state, message)

    async def decide_tool_call(
        self,
        session_id: SessionId,
        run_id: RunId,
        tool_call_id: str,
        approved: bool,
    ) -> RunState:
        if (
            type(run_id) is not RunId
            or type(tool_call_id) is not str
            or not tool_call_id
            or type(approved) is not bool
        ):
            raise SessionOwnershipError("Approval identity is invalid")
        slot = await self._slot(session_id)
        async with slot.gate.enter():
            async with self._operation(session_id):
                state = await self._head_state(session_id)
                call_id = _pending_call(state, session_id, run_id)
                if call_id != tool_call_id:
                    raise SessionOwnershipError("Tool call is not owned by this Session and Run")
                message = await self._user_input(state)
                coordinator = self._coordinator_factory(state, message)
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
                resumed = await self._coordinator_factory(decided, message).advance(
                    decided,
                    ResumeRun(
                        decided.run_id,
                        decided.session_id,
                        context=self._context(decided),
                    ),
                )
                return await self._drive(_outcome_state(resumed, decided), message)

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
            tasks = tuple(task for task in self._active if task is not current)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
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
        key = _session_key(session_id)
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
    async def _operation(self, session_id: SessionId):
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Application operation requires an asyncio Task")
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise ApplicationClosedError("Runtime is closed")
            self._active[task] = session_id.value
        try:
            yield
        finally:
            async with self._lifecycle_lock:
                self._active.pop(task, None)

    async def _cancel_tasks(self, session_key: str) -> None:
        current = asyncio.current_task()
        async with self._lifecycle_lock:
            tasks = tuple(
                task
                for task, key in self._active.items()
                if key == session_key and task is not current
            )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
        value = await self._journal.load_session(state.session_id)
        try:
            snapshot = snapshot_journal_snapshot(value)
        except (AttributeError, TypeError, ValueError):
            raise SessionOwnershipError("Session transcript is invalid")
        if (
            snapshot.session_id != state.session_id
            or snapshot.version != state.journal_version
            or snapshot.sequence != state.journal_sequence
            or snapshot.state is None
            or snapshot.state.run_id != state.run_id
        ):
            raise SessionOwnershipError("Session transcript is not the durable Run head")
        matches = [
            entry.content
            for entry in snapshot.transcript
            if entry.run_id == state.run_id
            and entry.submission_id == state.submission_id
            and entry.role == "user"
        ]
        if len(matches) != 1:
            raise SessionOwnershipError("Run does not have one durable user input")
        return matches[0]

    async def _drive(self, state: RunState, message: str) -> RunState:
        for _ in range(self._max_advances):
            if state.status.terminal or state.status is RunStatus.WAITING_APPROVAL:
                return state
            coordinator = self._coordinator_factory(state, message)
            outcome = await coordinator.advance(
                state,
                AdvanceRun(
                    state.run_id,
                    state.session_id,
                    context=self._context(state),
                ),
            )
            next_state = _outcome_state(outcome, state)
            if next_state.version <= state.version:
                raise RuntimeError("Run coordinator did not advance the state")
            state = next_state
        raise RuntimeError("Application operation exceeded its advance limit")

    def _context(self, state: RunState) -> CommandContext:
        return CommandContext(self._clock.now(), state.elapsed_seconds, (self._ids.event(),))


def _session_key(session_id: SessionId) -> str:
    if type(session_id) is not SessionId or type(session_id.value) is not str:
        raise SessionOwnershipError("Session identity is invalid")
    return SessionId(session_id.value).value


def _validate_submit(
    session_id: SessionId,
    profile_id: ProfileId,
    message: str,
    submission_id: SubmissionId,
) -> None:
    if (
        type(profile_id) is not ProfileId
        or type(submission_id) is not SubmissionId
        or type(message) is not str
    ):
        raise ValueError("Submission arguments are invalid")
    _session_key(session_id)
    ProfileId(profile_id.value)
    SubmissionId(submission_id.value)


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
