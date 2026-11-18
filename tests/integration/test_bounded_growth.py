from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.tokenizers import ExactTokenCounter
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite_codec import encode
from src.application.composition import ApplicationConfig, create_development_agent
from src.core.commands import RunLimits
from src.core.context import (
    ContextBudget,
    ContextRequest,
    ConversationInteraction,
    ConversationPage,
)
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.model import ModelMessage, ModelResponse, ModelRole


class Ids:
    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._value = 0

    def _next(self, prefix: str) -> str:
        self._value += 1
        return f"{self._namespace}-{prefix}-{self._value}"

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


class Tools:
    async def execute(self, _request):
        raise AssertionError("bounded direct answer must not execute a tool")


class PagedJournalReader:
    def __init__(self, interactions: tuple[ConversationInteraction, ...]) -> None:
        self._newest_first = tuple(reversed(interactions))
        self.calls: list[tuple[int | None, int, int]] = []

    async def read_page(
        self, session_id: SessionId, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        assert session_id == SessionId("bounded-session")
        eligible = tuple(
            item
            for item in self._newest_first
            if before_sequence is None or item.sequence_through < before_sequence
        )
        selected = eligible[:limit]
        next_cursor = None if len(eligible) <= limit else selected[-1].sequence_from
        self.calls.append((before_sequence, limit, len(selected)))
        return ConversationPage(selected, next_cursor)


class CapturingModel:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse("bounded-answer")


def _history(count: int) -> tuple[ConversationInteraction, ...]:
    return tuple(
        ConversationInteraction(
            index * 2 + 1,
            index * 2 + 2,
            (
                ModelMessage(ModelRole.USER, f"historical-user-{index}-" + "u" * 80),
                ModelMessage(ModelRole.ASSISTANT, f"historical-answer-{index}-" + "a" * 80),
            ),
        )
        for index in range(count)
    )


def test_ten_thousand_transcript_entries_are_paged_and_model_state_stays_bounded():
    async def scenario():
        reader = PagedJournalReader(_history(10_000))
        provider = BoundedContextProvider(
            reader,
            ExactTokenCounter(),
            page_size=127,
            max_compaction_interactions=32,
        )
        model = CapturingModel()
        journal = InMemoryRunJournal()
        repository = InMemoryArtifactRepository(1_048_576)
        externalizer = PolicyArtifactExternalizer(
            repository,
            ArtifactPolicy(128, 1_048_576, 256, frozenset({"text/plain"})),
        )
        skills = tuple(
            ModelMessage(ModelRole.SYSTEM, f"skill-{index}-" + "s" * 96) for index in range(128)
        )

        def request_factory(state, message):
            return ContextRequest(
                state.session_id,
                state.profile_id,
                "bounded-v1",
                ModelMessage(ModelRole.SYSTEM, "bounded system"),
                ModelMessage(ModelRole.USER, message),
                skill_messages=skills,
            )

        config = ApplicationConfig(
            limits=RunLimits(8, 30.0, 128),
            context_budget=ContextBudget(1_024, 128, 64),
        )
        agent = await create_development_agent(
            config,
            model=model,
            tools=Tools(),
            context=provider,
            context_request_factory=request_factory,
            ids=Ids("bounded"),
            clock=Clock(),
            profile_id=ProfileId("profile"),
            journal=journal,
            artifacts=externalizer,
        )
        session = await agent.open_session(SessionId("bounded-session"))
        result = await session.submit("current", SubmissionId("bounded-submission"))
        await agent.close()
        return config, reader, model, result

    config, reader, model, result = asyncio.run(scenario())
    assert len(reader.calls) >= 79
    assert all(limit == 127 and returned <= 127 for _, limit, returned in reader.calls)
    assert reader.calls[0][0] is None
    assert all(
        later[0] is not None and later[0] < earlier[0]
        for earlier, later in zip(reader.calls[1:], reader.calls[2:])
    )
    request = model.requests[0]
    estimated = ExactTokenCounter().count(request.messages, request.tools)
    assert estimated.tokens <= config.context_budget.input_allowance_tokens
    assert len(request.messages) < 32
    assert len(encode(result).encode("utf-8")) <= 4_096 + config.limits.max_inline_result_bytes
    assert "historical-user-0" not in encode(result)
