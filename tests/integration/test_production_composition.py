from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from src.adapters.artifacts.filesystem import FilesystemArtifactRepository
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.tokenizers import ExactTokenCounter
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.artifact_retention import ArtifactRetentionBlocked
from src.application.composition import ApplicationConfig, create_production_agent
from src.core.artifacts.contracts import ArtifactWrite
from src.core.context.contracts import ContextBudget, ContextRequest
from src.core.context.model import ModelMessage, ModelResponse, ModelRole, ModelToolCall
from src.core.execution.commands import RunLimits
from src.core.execution.state import RunStatus
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from tests.unit.application.test_agent_v3 import _valid_composition_dependencies


class _Ids:
    def __init__(self) -> None:
        self.value = 0

    def _next(self, kind: str) -> str:
        self.value += 1
        return f"production-{kind}-{self.value}"

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


class _Clock:
    def now(self):
        return datetime.now(timezone.utc)

    def monotonic(self):
        return time.monotonic()


class _Collector:
    async def collect_orphans(self, referenced_ids: tuple[str, ...], older_than: datetime) -> int:
        return 0


async def _chunks(data: bytes):
    yield data


def test_production_builder_wires_read_retention_gc_and_durable_tool_store(tmp_path) -> None:
    async def scenario():
        journal = SQLiteRunJournal(tmp_path / "runs.sqlite3")
        repository = FilesystemArtifactRepository(tmp_path / "artifacts", 1_048_576)
        source = await repository.put(ArtifactWrite(_chunks(b"production-artifact"), "text/plain"))
        orphan = await repository.put(ArtifactWrite(_chunks(b"orphan"), "text/plain"))

        class Model:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request):
                self.calls += 1
                assert any(spec.name == "read_artifact" for spec in request.tools)
                if self.calls == 1:
                    return ModelResponse(
                        tool_calls=(
                            ModelToolCall(
                                "read-call",
                                "read_artifact",
                                json.dumps(
                                    {
                                        "artifact_id": source.artifact_id,
                                        "offset": 0,
                                        "limit": 64,
                                    },
                                    separators=(",", ":"),
                                ),
                            ),
                        )
                    )
                transcript = "\n".join(message.content or "" for message in request.messages)
                assert "production-artifact" in transcript
                return ModelResponse("final-" + "x" * 2_048)

        model = Model()
        counter = ExactTokenCounter()

        def request_factory(state, message):
            return ContextRequest(
                state.session_id,
                state.profile_id,
                "production-v1",
                ModelMessage(ModelRole.SYSTEM, "production composition"),
                ModelMessage(ModelRole.USER, message),
            )

        agent = await create_production_agent(
            ApplicationConfig(
                limits=RunLimits(16, 30.0, 512),
                context_budget=ContextBudget(8_192, 512, 256),
                sqlite_path=tmp_path / "runs.sqlite3",
                tool_execution_path=tmp_path / "tool-executions.sqlite3",
                artifact_root=tmp_path / "artifacts",
                artifact_workspace_id="production-artifacts",
                max_inline_artifact_bytes=512,
                max_artifact_summary_bytes=256,
                max_artifact_read_bytes=64,
                max_orphan_reference_ids=1_024,
            ),
            model=model,
            tools=None,
            context=BoundedContextProvider(journal, counter),
            context_counter=counter,
            context_request_factory=request_factory,
            ids=_Ids(),
            clock=_Clock(),
            profile_id=ProfileId("profile"),
            journal=journal,
            artifact_repository=repository,
        )
        session = await agent.open_session(SessionId("production-session"))
        state = await session.submit("read it", SubmissionId("production-submission"))
        assert state.status is RunStatus.COMPLETED
        assert model.calls == 2
        final = state.final_artifacts[0]

        with pytest.raises(ArtifactRetentionBlocked):
            await agent.artifacts.retire(final.artifact_id)
        await session.detach_artifacts(state.run_id)
        await agent.artifacts.retire(final.artifact_id)
        assert await repository.stat(final.artifact_id) is None

        collected = await agent.artifacts.collect_orphans(
            datetime.now(timezone.utc) + timedelta(days=1)
        )
        assert collected == 2
        assert await repository.stat(source.artifact_id) is None
        assert await repository.stat(orphan.artifact_id) is None
        await agent.close()

    asyncio.run(scenario())


def test_production_config_rejects_shared_sqlite_and_unsafe_read_limits(tmp_path) -> None:
    database = tmp_path / "shared.sqlite3"
    with pytest.raises(ValueError, match="configuration limits"):
        ApplicationConfig(
            sqlite_path=database,
            tool_execution_path=database,
            artifact_root=tmp_path / "artifacts",
        )

    async def scenario() -> None:
        journal = SQLiteRunJournal(tmp_path / "runs.sqlite3")
        repository = FilesystemArtifactRepository(tmp_path / "artifacts", 1_048_576)
        counter = ExactTokenCounter()

        class Model:
            async def complete(self, _request):
                raise AssertionError("unsafe production configuration reached the model")

        def request_factory(state, message):
            return ContextRequest(
                state.session_id,
                state.profile_id,
                "production-v1",
                ModelMessage(ModelRole.SYSTEM, "production composition"),
                ModelMessage(ModelRole.USER, message),
            )

        with pytest.raises(ValueError, match="Production Artifact limits"):
            await create_production_agent(
                ApplicationConfig(
                    limits=RunLimits(8, 30.0, 32),
                    context_budget=ContextBudget(512, 32, 16),
                    sqlite_path=tmp_path / "runs.sqlite3",
                    artifact_root=tmp_path / "artifacts",
                    max_inline_artifact_bytes=32,
                    max_artifact_summary_bytes=32,
                    max_artifact_read_bytes=33,
                ),
                model=Model(),
                tools=None,
                context=BoundedContextProvider(journal, counter),
                context_counter=counter,
                context_request_factory=request_factory,
                ids=_Ids(),
                clock=_Clock(),
                profile_id=ProfileId("profile"),
                journal=journal,
                artifact_repository=repository,
            )

    asyncio.run(scenario())


def test_injected_remote_artifacts_need_workspace_identity_not_local_root(tmp_path) -> None:
    async def scenario() -> None:
        agent = await create_production_agent(
            ApplicationConfig(
                artifact_workspace_id="remote-artifacts",
                tool_execution_path=(tmp_path / "tools.sqlite3").resolve(),
            ),
            journal=InMemoryRunJournal(),
            artifact_repository=InMemoryArtifactRepository(1_048_576),
            artifact_collector=_Collector(),
            **_valid_composition_dependencies(),
        )
        await agent.close()

    asyncio.run(scenario())


def test_production_rejects_split_atomic_artifact_authorities(tmp_path) -> None:
    journal = InMemoryRunJournal()
    with pytest.raises(ValueError, match="same atomic persistence authority"):
        asyncio.run(
            create_production_agent(
                ApplicationConfig(
                    artifact_workspace_id="remote-artifacts",
                    tool_execution_path=(tmp_path / "tools.sqlite3").resolve(),
                ),
                journal=journal,
                artifact_ownership=InMemoryRunJournal(),
                artifact_repository=InMemoryArtifactRepository(1_048_576),
                artifact_collector=_Collector(),
                **_valid_composition_dependencies(),
            )
        )


@pytest.mark.parametrize("workspace_id", ("", "../escape", "a" * 257))
def test_production_rejects_invalid_artifact_workspace_identity(workspace_id: str) -> None:
    with pytest.raises(ValueError, match="workspace"):
        ApplicationConfig(artifact_workspace_id=workspace_id)
