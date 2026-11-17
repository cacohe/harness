from __future__ import annotations

import asyncio

from src.application.runtime import RuntimeCoordinator
from src.core.ids import IdGenerator, ProfileId, RunId, SessionId, SubmissionId
from src.core.state import RunState


class AgentSession:
    """An async Session handle; serialization remains owned by the runtime."""

    def __init__(
        self,
        runtime: RuntimeCoordinator,
        ids: IdGenerator,
        session_id: SessionId,
        profile_id: ProfileId,
    ) -> None:
        self._runtime = runtime
        self._ids = ids
        self.session_id = SessionId(session_id.value)
        self.profile_id = ProfileId(profile_id.value)
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def submit(self, message: str, submission_id: SubmissionId | None = None) -> RunState:
        self._ensure_open()
        identifier = self._ids.submission() if submission_id is None else submission_id
        return await self._runtime.submit(self.session_id, self.profile_id, message, identifier)

    async def decide_tool_call(self, run_id: RunId, tool_call_id: str, approved: bool) -> RunState:
        self._ensure_open()
        return await self._runtime.decide_tool_call(self.session_id, run_id, tool_call_id, approved)

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            self._closed = True
            task = asyncio.create_task(self._runtime.close_session(self.session_id))
            self._close_task = task
        await asyncio.shield(task)

    async def __aenter__(self) -> AgentSession:
        self._ensure_open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent Session is closed")


class Agent:
    """Async-first application facade with no implicit production dependencies."""

    def __init__(
        self,
        runtime: RuntimeCoordinator,
        ids: IdGenerator,
        profile_id: ProfileId,
        *,
        default_session_id: SessionId | None = None,
    ) -> None:
        if type(profile_id) is not ProfileId or (
            default_session_id is not None and type(default_session_id) is not SessionId
        ):
            raise ValueError("Agent identities are invalid")
        self._runtime = runtime
        self._ids = ids
        self.profile_id = ProfileId(profile_id.value)
        self._default_session_id = (
            SessionId(default_session_id.value) if default_session_id is not None else None
        )
        self._default_session: AgentSession | None = None
        self._default_lock = asyncio.Lock()
        self._closed = False

    async def open_session(
        self,
        session_id: SessionId | None = None,
        *,
        profile_id: ProfileId | None = None,
    ) -> AgentSession:
        self._ensure_open()
        identifier = self._ids.session() if session_id is None else session_id
        profile = self.profile_id if profile_id is None else profile_id
        if type(identifier) is not SessionId or type(profile) is not ProfileId:
            raise ValueError("Agent Session identities are invalid")
        await self._runtime.open_session(identifier)
        return AgentSession(self._runtime, self._ids, identifier, profile)

    async def submit(self, message: str, submission_id: SubmissionId | None = None) -> RunState:
        session = await self._get_default_session()
        return await session.submit(message, submission_id)

    async def close(self) -> None:
        self._closed = True
        await self._runtime.close()

    async def __aenter__(self) -> Agent:
        self._ensure_open()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def _get_default_session(self) -> AgentSession:
        self._ensure_open()
        async with self._default_lock:
            if self._default_session is None:
                self._default_session = await self.open_session(self._default_session_id)
            return self._default_session

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent is closed")
