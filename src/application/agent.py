from __future__ import annotations

import asyncio

from src.application.artifact_retention import ArtifactRetentionService
from src.application.runtime import RuntimeCoordinator
from src.core.artifacts.lifecycle import ArtifactAttachmentFact, ArtifactAttachmentRole
from src.core.execution.state import RunState
from src.core.ids import IdGenerator, ProfileId, RunId, SessionId, SubmissionId


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
        self.session_id = _canonical_id(session_id, SessionId)
        self.profile_id = _canonical_id(profile_id, ProfileId)
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def submit(self, message: str, submission_id: SubmissionId | None = None) -> RunState:
        self._ensure_open()
        identifier = self._ids.submission() if submission_id is None else submission_id
        return await self._runtime.submit(self.session_id, self.profile_id, message, identifier)

    async def decide_tool_call(self, run_id: RunId, tool_call_id: str, approved: bool) -> RunState:
        self._ensure_open()
        return await self._runtime.decide_tool_call(self.session_id, run_id, tool_call_id, approved)

    async def cancel(self, run_id: RunId) -> RunState:
        self._ensure_open()
        return await self._runtime.cancel_run(self.session_id, run_id)

    async def current_run(self) -> RunState | None:
        self._ensure_open()
        return await self._runtime.current_run(self.session_id)

    async def detach_artifacts(
        self,
        run_id: RunId,
        *,
        roles: tuple[ArtifactAttachmentRole, ...] | None = None,
    ) -> tuple[ArtifactAttachmentFact, ...]:
        self._ensure_open()
        return await self._runtime.detach_artifacts(self.session_id, run_id, roles=roles)

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
        artifacts: ArtifactRetentionService | None = None,
    ) -> None:
        self._runtime = runtime
        self._ids = ids
        self.profile_id = _canonical_id(profile_id, ProfileId)
        self._default_session_id = (
            _canonical_id(default_session_id, SessionId) if default_session_id is not None else None
        )
        self._default_session: AgentSession | None = None
        self._artifact_services = artifacts
        self._default_lock = asyncio.Lock()
        self._closed = False

    @property
    def artifacts(self) -> ArtifactRetentionService:
        if self._artifact_services is None:
            raise RuntimeError("Artifact maintenance is not configured")
        return self._artifact_services

    async def open_session(
        self,
        session_id: SessionId | None = None,
        *,
        profile_id: ProfileId | None = None,
    ) -> AgentSession:
        self._ensure_open()
        identifier = self._ids.session() if session_id is None else session_id
        profile = self.profile_id if profile_id is None else profile_id
        identifier = _canonical_id(identifier, SessionId)
        profile = _canonical_id(profile, ProfileId)
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


def _canonical_id(value, expected):
    try:
        if type(value) is not expected:
            raise ValueError
        raw = object.__getattribute__(value, "value")
        if type(raw) is not str:
            raise ValueError
        return expected(raw)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Agent identities are invalid") from None
