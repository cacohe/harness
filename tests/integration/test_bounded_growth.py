from __future__ import annotations

import asyncio
import json
import time
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.tokenizers import ExactTokenCounter
from src.adapters.persistence.bounded import BoundedRunJournal
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.adapters.persistence.sqlite_codec import encode
from src.application.composition import ApplicationConfig, create_development_agent
from src.core.artifacts import ArtifactRef
from src.core.commands import RunLimits
from src.core.context import (
    ContextBudget,
    ContextRequest,
    ConversationCoverage,
    ConversationCoverageSegment,
    ConversationInteraction,
    ConversationPage,
    OmittedContextRef,
)
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalError,
    TranscriptEntry,
)
from src.core.model import ModelMessage, ModelResponse, ModelRole, ModelToolCall
from src.core.tools import ToolResult, ToolResultStatus, ToolSpec
from tests.unit.core.test_journal_contract import acceptance


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
        self._interactions = interactions
        self.calls: list[tuple[int | None, int, int]] = []
        self.materialized_interactions = 0
        self.peak_materialized_interactions = 0
        self.peak_materialized_bytes = 0

    async def read_page(
        self, session_id: SessionId, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        assert session_id == SessionId("bounded-session")
        end = (
            len(self._interactions)
            if before_sequence is None
            else bisect_left(
                self._interactions,
                before_sequence,
                key=lambda item: item.sequence_through,
            )
        )
        start = max(0, end - limit)
        selected = tuple(reversed(self._interactions[start:end]))
        self.materialized_interactions += len(selected)
        self.peak_materialized_interactions = max(
            self.peak_materialized_interactions, len(selected)
        )
        self.peak_materialized_bytes = max(
            self.peak_materialized_bytes,
            sum(
                len((message.content or "").encode("utf-8"))
                for item in selected
                for message in item.messages
            ),
        )
        has_more = start > 0
        next_cursor = selected[-1].sequence_from if has_more else None
        remaining = None
        if has_more:
            remaining = OmittedContextRef(
                "transcript",
                self._interactions[0].sequence_from,
                self._interactions[start - 1].sequence_through,
                start,
            )
        self.calls.append((before_sequence, limit, len(selected)))
        return ConversationPage(selected, next_cursor, remaining_omission=remaining)

    async def summarize_before(
        self,
        session_id,
        before_sequence,
        excluded_from_sequence=None,
        excluded_through_sequence=None,
    ):
        assert session_id == SessionId("bounded-session")
        target_last = before_sequence - 1
        ranges = []
        if target_last > 0:
            if excluded_from_sequence is None or excluded_from_sequence > target_last:
                ranges.append((1, target_last))
            else:
                assert excluded_through_sequence is not None
                if excluded_through_sequence < target_last:
                    ranges.append((excluded_through_sequence + 1, target_last))
                if excluded_from_sequence > 1:
                    ranges.append((1, excluded_from_sequence - 1))
        segments = tuple(
            ConversationCoverageSegment(
                first,
                last,
                bisect_left(
                    self._interactions,
                    last + 1,
                    key=lambda item: item.sequence_through,
                )
                - bisect_left(
                    self._interactions,
                    first,
                    key=lambda item: item.sequence_through,
                ),
            )
            for first, last in ranges
        )
        return ConversationCoverage(
            before_sequence,
            excluded_from_sequence,
            excluded_through_sequence,
            segments,
        )


class CountingJournalReader:
    def __init__(self, journal) -> None:
        self.journal = journal
        self.page_calls: list[tuple[int | None, int, int]] = []
        self.aggregate_calls: list[tuple[int, int | None, int | None]] = []

    async def read_page(self, session_id, before_sequence, limit):
        page = await self.journal.read_page(session_id, before_sequence, limit)
        self.page_calls.append((before_sequence, limit, len(page.interactions)))
        return page

    async def summarize_before(
        self,
        session_id,
        before_sequence,
        excluded_from_sequence=None,
        excluded_through_sequence=None,
    ):
        self.aggregate_calls.append(
            (before_sequence, excluded_from_sequence, excluded_through_sequence)
        )
        return await self.journal.summarize_before(
            session_id,
            before_sequence,
            excluded_from_sequence,
            excluded_through_sequence,
        )


class CapturingModel:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse("bounded-answer")


class RecordingJournal(InMemoryRunJournal):
    def __init__(self) -> None:
        super().__init__()
        self.state_sizes: list[int] = []

    async def commit(self, batch, expected_version):
        self.state_sizes.append(len(encode(batch.state).encode("utf-8")))
        return await super().commit(batch, expected_version)


class BatchModel:
    async def complete(self, request):
        payload = json.loads(request.messages[-1].content)
        if payload["round"] == 0:
            return ModelResponse(
                tool_calls=tuple(
                    ModelToolCall(
                        f"bulk-call-{index}",
                        "bulk",
                        json.dumps(
                            {"index": index, "payload": "a" * 2_048},
                            separators=(",", ":"),
                        ),
                    )
                    for index in range(12)
                )
            )
        return ModelResponse("bounded batch complete")


class ArtifactTools:
    async def execute(self, request):
        index = int(request.tool_call_id.rsplit("-", 1)[1])
        digest = f"{index + 1:064x}"
        return ToolResult(
            ToolResultStatus.SUCCESS,
            artifacts=(
                ArtifactRef(
                    "sha256:" + digest,
                    "application/json",
                    2_048,
                    digest,
                    metadata={"owner": request.session_id.value},
                ),
            ),
            summary=f"bounded artifact result {index}",
            execution_key=request.idempotency_key,
        )


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
            context_counter=ExactTokenCounter(),
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
    assert len(reader.calls) <= 2
    assert all(limit == 127 and returned <= 127 for _, limit, returned in reader.calls)
    assert reader.calls[0][0] is None
    assert reader.materialized_interactions <= 254
    assert reader.peak_materialized_interactions <= 127
    assert reader.peak_materialized_bytes <= 32_000
    request = model.requests[0]
    estimated = ExactTokenCounter().count(request.messages, request.tools)
    assert estimated.tokens <= config.context_budget.input_allowance_tokens
    assert len(request.messages) < 32
    assert len(encode(result).encode("utf-8")) <= 4_096 + config.limits.max_inline_result_bytes
    assert "historical-user-0" not in encode(result)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
def test_production_journal_reader_bounds_partial_checkpoint_history(
    backend: str, tmp_path: Path
) -> None:
    async def scenario():
        journal = (
            InMemoryRunJournal()
            if backend == "memory"
            else SQLiteRunJournal((tmp_path / "bounded-history.sqlite3").resolve())
        )
        batch = acceptance("bounded-history")
        transcript = tuple(
            TranscriptEntry(
                batch.state.session_id,
                batch.state.run_id,
                batch.state.submission_id,
                "assistant",
                f"history-{index:05d}",
            )
            for index in range(10_000)
        )
        checkpoint = ConversationSummaryCheckpoint(
            batch.state.session_id,
            6_000,
            "v1",
            batch.state.profile_id,
            "durable middle summary",
            covered_from_sequence=4_001,
        )
        for chunk in range(10):
            commit = replace(
                batch,
                state=replace(
                    batch.state,
                    journal_version=chunk + 1,
                    journal_sequence=1 + (chunk + 1) * 1_000,
                ),
                events=batch.events if chunk == 0 else (),
                submission=None,
                transcript=transcript[chunk * 1_000 : (chunk + 1) * 1_000],
                summary=checkpoint if chunk == 9 else None,
            )
            await journal.commit(commit, chunk)
        reader = CountingJournalReader(journal)
        provider = BoundedContextProvider(
            reader,
            ExactTokenCounter(),
            page_size=127,
            max_compaction_interactions=128,
        )
        request = ContextRequest(
            batch.state.session_id,
            batch.state.profile_id,
            "v1",
            ModelMessage(ModelRole.SYSTEM, "bounded production history"),
            ModelMessage(ModelRole.USER, "current"),
            summary_checkpoints=(checkpoint,),
        )
        window = await provider.build(request, ContextBudget(128, 8, 8))
        return reader, window

    reader, window = asyncio.run(scenario())
    assert len(reader.page_calls) <= 2
    assert all(limit == 127 and returned <= 127 for _, limit, returned in reader.page_calls)
    assert reader.aggregate_calls == [(9_748, 4_001, 6_000)]
    transcript_omissions = tuple(item for item in window.omitted if item.source == "transcript")
    assert transcript_omissions[0].first_sequence == 1
    assert transcript_omissions[0].last_sequence == 4_000
    assert transcript_omissions[1].first_sequence == 6_001
    selected = tuple(
        message
        for message in window.messages
        if message.role is ModelRole.ASSISTANT and (message.content or "").startswith("history-")
    )
    assert transcript_omissions[1].last_sequence == 10_001 - len(selected)
    assert sum(item.item_count for item in transcript_omissions) + 2_000 + len(selected) == 10_000


def test_configured_serialized_run_envelope_bounds_large_batch_and_artifact_results():
    async def scenario(max_state_bytes: int):
        reader = PagedJournalReader(_history(10_000))
        context = BoundedContextProvider(
            reader,
            ExactTokenCounter(),
            page_size=127,
            max_compaction_interactions=16,
        )
        journal = RecordingJournal()
        repository = InMemoryArtifactRepository(1_048_576)
        externalizer = PolicyArtifactExternalizer(
            repository,
            ArtifactPolicy(256, 1_048_576, 128, frozenset({"text/plain"})),
        )
        tool = ToolSpec("bulk", "bounded batch tool", "{}")
        skills = tuple(
            ModelMessage(ModelRole.SYSTEM, f"large-skill-{index}-" + "s" * 128)
            for index in range(128)
        )

        def request_factory(state, message):
            return ContextRequest(
                state.session_id,
                state.profile_id,
                "state-envelope-v1",
                ModelMessage(ModelRole.SYSTEM, "state envelope"),
                ModelMessage(
                    ModelRole.USER,
                    json.dumps(
                        {"message": message, "round": len(state.used_step_ids)},
                        separators=(",", ":"),
                    ),
                ),
                (tool,),
                skill_messages=skills,
            )

        config = ApplicationConfig(
            limits=RunLimits(32, 30.0, 256),
            context_budget=ContextBudget(4_096, 256, 128),
            max_serialized_run_state_bytes=max_state_bytes,
        )
        agent = await create_development_agent(
            config,
            model=BatchModel(),
            tools=ArtifactTools(),
            context=context,
            context_counter=ExactTokenCounter(),
            context_request_factory=request_factory,
            ids=Ids(f"state-{max_state_bytes}"),
            clock=Clock(),
            profile_id=ProfileId("profile"),
            journal=journal,
            artifacts=externalizer,
        )
        session = await agent.open_session(SessionId("bounded-session"))
        try:
            state = await session.submit("large batch", SubmissionId("large-batch"))
        finally:
            await agent.close()
        return config, journal, state

    config, journal, state = asyncio.run(scenario(48 * 1_024))
    assert isinstance(
        BoundedRunJournal(journal, config.max_serialized_run_state_bytes), BoundedRunJournal
    )
    assert max(journal.state_sizes) <= config.max_serialized_run_state_bytes
    assert max(journal.state_sizes) >= 24 * 1_024
    assert state.final_text == "bounded batch complete"
    assert state.current_step is None

    with pytest.raises(JournalError) as rejected:
        asyncio.run(scenario(8 * 1_024))
    assert rejected.value.failure.code.value == "limit_exceeded"
