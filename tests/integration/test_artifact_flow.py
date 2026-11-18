from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone

from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.filesystem import FilesystemArtifactRepository
from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.tokenizers import ExactTokenCounter
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite_codec import encode
from src.adapters.tools import (
    InMemoryToolExecutionStore,
    LocalToolGateway,
    ReadArtifactTool,
    ToolCatalog,
    ToolCatalogEntry,
)
from src.application.composition import ApplicationConfig, create_development_agent
from src.core.commands import RunLimits
from src.core.context import (
    ContextBudget,
    ContextRequest,
)
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.model import ModelMessage, ModelResponse, ModelRole, ModelToolCall
from src.core.state import RunStatus
from src.core.tools import ToolSpec


class Ids:
    def __init__(self) -> None:
        self._value = 0

    def _next(self, prefix: str) -> str:
        self._value += 1
        return f"artifact-{prefix}-{self._value}"

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
        return datetime.now(timezone.utc)

    def monotonic(self):
        return time.monotonic()


class JournalContext:
    def __init__(self, journal: InMemoryRunJournal) -> None:
        self._provider = BoundedContextProvider(journal, ExactTokenCounter())

    async def build(self, request: ContextRequest, budget: ContextBudget):
        return await self._provider.build(request, budget)


class ArtifactModel:
    def __init__(self) -> None:
        self.requests = []
        self.artifact_id: str | None = None
        self.final_body = "final-" + "z" * 8_192

    async def complete(self, request):
        self.requests.append(request)
        payload = json.loads(request.messages[-1].content)
        if payload["message"] == "after-final":
            return ModelResponse("follow-up")
        round_number = payload["round"]
        if round_number == 0:
            return ModelResponse(tool_calls=(ModelToolCall("produce-call", "produce", "{}"),))
        history = "\n".join(message.content or "" for message in request.messages[:-1])
        match = re.search(r"sha256:[0-9a-f]{64}", history)
        if round_number == 1:
            assert match is not None
            self.artifact_id = match.group(0)
            return ModelResponse(
                tool_calls=(
                    ModelToolCall(
                        "read-call",
                        "read_artifact",
                        json.dumps(
                            {"artifact_id": self.artifact_id, "offset": 0, "limit": 32},
                            separators=(",", ":"),
                        ),
                    ),
                )
            )
        assert '"returned_bytes":32' in history
        assert '"truncated":true' in history
        return ModelResponse(self.final_body)


def test_large_tool_result_is_referenced_read_in_bounded_slice_and_never_journaled(tmp_path):
    async def scenario():
        root = tmp_path / "artifact-store"
        repository = FilesystemArtifactRepository(root, 1_048_576)
        policy = ArtifactPolicy(
            512,
            1_048_576,
            192,
            frozenset({"application/json", "text/plain"}),
        )
        externalizer = PolicyArtifactExternalizer(repository, policy)
        large_body = "artifact-secret-body-" + "x" * 8_192

        async def produce(_arguments):
            return {"payload": large_body, "count": 1}

        read_artifact = ReadArtifactTool(repository, max_read_bytes=64)
        produce_spec = ToolSpec("produce", "produce a large result", "{}")
        read_spec = ToolSpec(
            "read_artifact",
            "read one bounded artifact range",
            '{"type":"object","required":["artifact_id","offset","limit"]}',
        )
        catalog = ToolCatalog(
            (
                ToolCatalogEntry(produce_spec, produce, False),
                ToolCatalogEntry(read_spec, read_artifact, False),
            )
        )
        gateway = LocalToolGateway(
            catalog,
            InMemoryToolExecutionStore(),
            externalizer,
            timeout_seconds=2,
        )
        journal = InMemoryRunJournal()
        model = ArtifactModel()

        def request_factory(state, message):
            return ContextRequest(
                state.session_id,
                state.profile_id,
                "artifact-v1",
                ModelMessage(ModelRole.SYSTEM, "artifact integration"),
                ModelMessage(
                    ModelRole.USER,
                    json.dumps(
                        {"message": message, "round": len(state.used_step_ids)},
                        separators=(",", ":"),
                    ),
                ),
                catalog.specs,
            )

        agent = await create_development_agent(
            ApplicationConfig(
                limits=RunLimits(16, 30.0, 512),
                context_budget=ContextBudget(8_192, 512, 256),
            ),
            model=model,
            tools=gateway,
            context=JournalContext(journal),
            context_counter=ExactTokenCounter(),
            context_request_factory=request_factory,
            ids=Ids(),
            clock=Clock(),
            profile_id=ProfileId("profile"),
            journal=journal,
            artifacts=externalizer,
        )
        session = await agent.open_session(SessionId("artifact-session"))
        result = await session.submit("artifact-flow", SubmissionId("artifact-submission"))
        follow_up = await session.submit("after-final", SubmissionId("after-final"))
        snapshot = await journal.load_session(SessionId("artifact-session"))
        assert model.artifact_id is not None
        source_ref = await repository.stat(model.artifact_id)
        final_bytes = bytearray()
        async for chunk in repository.open(result.final_artifacts[0].artifact_id):
            final_bytes.extend(chunk)
        await agent.close()
        return root, large_body, model, result, follow_up, snapshot, source_ref, bytes(final_bytes)

    root, large_body, model, result, follow_up, snapshot, source_ref, final_bytes = asyncio.run(
        scenario()
    )
    assert result.status is RunStatus.COMPLETED
    assert follow_up.status is RunStatus.COMPLETED
    assert source_ref is not None
    assert result.final_artifacts
    assert all(len(request.messages) < 16 for request in model.requests)
    second_wire = "\n".join(message.content or "" for message in model.requests[1].messages)
    assert source_ref.artifact_id in second_wire
    assert large_body not in second_wire
    journal_wire = encode(snapshot)
    run_wire = encode(result)
    transcript_wire = encode(snapshot.transcript)
    events_wire = encode(snapshot.events)
    assert source_ref.artifact_id in journal_wire
    assert large_body not in journal_wire
    assert model.final_body.encode("utf-8") == final_bytes
    assert model.final_body not in run_wire
    assert model.final_body not in transcript_wire
    assert model.final_body not in events_wire
    assert model.final_body not in journal_wire
    assert model.final_body not in "\n".join(
        message.content or "" for message in model.requests[-1].messages
    )
    assert len(result.final_text.encode("utf-8")) <= 512
    assert result.final_artifacts[0].artifact_id in journal_wire
    assert str(root) not in journal_wire
    assert all(ref.artifact_id != source_ref.artifact_id for ref in result.final_artifacts)
