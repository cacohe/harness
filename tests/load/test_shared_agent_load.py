import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone

from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.agent import Agent
from src.application.composition import ApplicationConfig, create_development_agent
from src.config import Config
from src.core.commands import RunLimits
from src.core.context import (
    ContextBudget,
    ContextRequest,
    ContextWindow,
    EstimateQuality,
    TokenEstimate,
)
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import ConversationSummaryCheckpoint
from src.core.memory import MemoryManager
from src.core.model import ModelMessage, ModelResponse, ModelRole
from src.memory.retriever import HybridRetriever
from src.memory.store import InMemoryStore
from src.observability import MetricsEventSink
from src.persistence.in_memory import InMemoryTaskRepository
from src.schema.llm import FinishReason, LLMOutput
from src.schema.task import TaskState


class YieldingGateway:
    """Deterministic async gateway that forces sessions to interleave."""

    def __init__(self, expected_calls=64):
        self.expected_calls = expected_calls
        self.infer_inputs = []
        self.infer_arrived = 0
        self.all_infer_arrived = asyncio.Event()

    @staticmethod
    def _current_message(context):
        content = context.messages[1].content or ""
        marker = "Current user message:\n"
        return content.split(marker, 1)[1] if marker in content else content

    async def ainfer(self, context, stream=True, response_format=None):
        del stream, response_format
        current = self._current_message(context)
        self.infer_inputs.append(current)
        self.infer_arrived += 1
        if self.infer_arrived == self.expected_calls:
            self.all_infer_arrived.set()
        await self.all_infer_arrived.wait()
        return LLMOutput(FinishReason.STOP, f"answer:{current}", [], None, None)


def _legacy_shared_agent_handles_concurrent_sessions_without_state_leaks():
    async def scenario():
        repository = InMemoryTaskRepository()
        memory = MemoryManager(InMemoryStore(), HybridRetriever())
        metrics = MetricsEventSink(max_inflight=1_000)
        gateway = YieldingGateway()
        agent = Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=gateway,
            task_repository=repository,
            memory_manager=memory,
            event_sink=metrics,
        )
        sessions = [agent.new_session() for _ in range(64)]
        tasks = await asyncio.wait_for(
            asyncio.gather(
                *(session.achat(f"load-{index}") for index, session in enumerate(sessions))
            ),
            timeout=10,
        )
        return repository, memory, metrics, gateway, sessions, tasks

    repository, memory, metrics, gateway, sessions, tasks = asyncio.run(scenario())

    expected_inputs = {f"load-{index}" for index in range(64)}
    assert set(gateway.infer_inputs) == expected_inputs
    assert gateway.infer_arrived == 64
    assert len({task.id for task in tasks}) == 64
    assert all(task.state is TaskState.TASK_COMPLETED for task in tasks)
    for index, (session, task) in enumerate(zip(sessions, tasks, strict=True)):
        expected = f"answer:load-{index}"
        assert task.result.output == expected
        assert session.transcript == [("user", f"load-{index}"), ("assistant", expected)]
        assert repository.get(task.id) == task

    items = memory.store.all()
    assert len(items) == 64
    memories_by_session = {item.metadata["session_id"]: item for item in items}
    assert len(memories_by_session) == 64
    for index, (session, task) in enumerate(zip(sessions, tasks, strict=True)):
        item = memories_by_session[session.id]
        assert item.metadata == {"session_id": session.id, "task_id": task.id}
        assert json.loads(item.content) == {
            "user": f"load-{index}",
            "assistant": f"answer:load-{index}",
        }

    snapshot = metrics.snapshot()
    assert snapshot["events"]["task_started"] == 64
    assert snapshot["events"]["task_completed"] == 64
    assert snapshot["events"]["llm_started"] == 64
    assert snapshot["events"]["llm_completed"] == 64
    assert snapshot["task_outcomes"] == {"completed": 64}
    assert snapshot["in_flight"] == {"tasks": 0, "llm": 0, "tools": 0}
    assert snapshot["inflight_evictions"] == {"tasks": 0, "llm": 0, "tools": 0}
    assert snapshot["durations"]["task"]["count"] == 64
    assert snapshot["durations"]["llm"]["count"] == 64


# V3 load contract. The legacy body above is physically removed in Task 14; this
# file cannot currently be replaced atomically on Windows because of inherited ACLs.


class V3Ids:
    def __init__(self):
        self._value = 0

    def _next(self, prefix):
        self._value += 1
        return f"load-{prefix}-{self._value}"

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


class V3Clock:
    def now(self):
        return datetime.now(timezone.utc)

    def monotonic(self):
        return time.monotonic()


class NoTools:
    async def execute(self, _request):
        raise AssertionError("load scenario does not execute tools")


class PerSessionContext:
    async def build(self, request, budget):
        checkpoint = ConversationSummaryCheckpoint(
            request.session_id,
            2,
            request.policy_version,
            request.profile_id,
            f"summary:{request.session_id.value}",
            2,
        )
        messages = (request.system_message, request.current_input)
        return ContextWindow(
            messages,
            request.tools,
            TokenEstimate(2, EstimateQuality.EXACT),
            budget,
            (),
            request.policy_version,
            checkpoint,
        )


class BarrierModel:
    def __init__(self, expected):
        self._expected = expected
        self._arrived = 0
        self._ready = asyncio.Event()
        self.calls = Counter()

    async def complete(self, request):
        payload = json.loads(request.messages[-1].content)
        session = payload["session"]
        message = payload["message"]
        self.calls[(session, message)] += 1
        if message.startswith("session-"):
            self._arrived += 1
            if self._arrived == self._expected:
                self._ready.set()
            await self._ready.wait()
        return ModelResponse(f"answer:{session}:{message}:" + "x" * 512)


def test_shared_agent_isolates_64_sessions_and_same_session_submission_race():
    async def scenario():
        count = 64
        journal = InMemoryRunJournal()
        repository = InMemoryArtifactRepository(8 * 1_048_576)
        externalizer = PolicyArtifactExternalizer(
            repository,
            ArtifactPolicy(64, 1_048_576, 64, frozenset({"text/plain"})),
        )
        model = BarrierModel(count)

        def request_factory(state, message):
            return ContextRequest(
                state.session_id,
                state.profile_id,
                "load-v1",
                ModelMessage(ModelRole.SYSTEM, "load system"),
                ModelMessage(
                    ModelRole.USER,
                    json.dumps(
                        {"session": state.session_id.value, "message": message},
                        separators=(",", ":"),
                    ),
                ),
            )

        agent = await create_development_agent(
            ApplicationConfig(
                max_sessions=65,
                max_waiters_per_session=2,
                limits=RunLimits(8, 120.0, 64),
                context_budget=ContextBudget(512, 32, 16),
            ),
            model=model,
            tools=NoTools(),
            context=PerSessionContext(),
            context_request_factory=request_factory,
            ids=V3Ids(),
            clock=V3Clock(),
            profile_id=ProfileId("profile"),
            journal=journal,
            artifacts=externalizer,
        )
        sessions = [
            await agent.open_session(SessionId(f"load-session-{index}")) for index in range(count)
        ]
        states = await asyncio.wait_for(
            asyncio.gather(
                *(
                    session.submit(f"session-{index}", SubmissionId(f"submission-{index}"))
                    for index, session in enumerate(sessions)
                )
            ),
            timeout=60,
        )
        duplicate_id = SubmissionId("same-submission")
        duplicate = await asyncio.gather(
            sessions[0].submit("race", duplicate_id),
            sessions[0].submit("race", duplicate_id),
        )
        snapshots = [
            await journal.load_session(SessionId(f"load-session-{index}")) for index in range(count)
        ]
        assert all(state.status.value == "completed" for state in states), [
            (state.status.value, state.failure) for state in states
        ]
        refs = [state.final_artifacts[0] for state in states]
        stored = await asyncio.gather(*(repository.stat(ref.artifact_id) for ref in refs))
        await agent.close()
        return model, states, duplicate, snapshots, refs, stored

    model, states, duplicate, snapshots, refs, stored = asyncio.run(scenario())
    assert len({state.run_id.value for state in states}) == 64
    assert len({ref.artifact_id for ref in refs}) == 64
    assert all(snapshot.state == state for snapshot, state in zip(snapshots[1:], states[1:]))
    assert all(
        snapshot.version == states[index].journal_version
        for index, snapshot in enumerate(snapshots[1:], start=1)
    )
    assert all(
        snapshot.summaries
        and snapshot.summaries[0].session_id == snapshot.session_id
        and snapshot.summaries[0].summary == f"summary:{snapshot.session_id.value}"
        for snapshot in snapshots
    )
    assert stored == refs
    assert duplicate[0] == duplicate[1]
    assert model.calls[("load-session-0", "race")] == 1
    assert snapshots[0].state == duplicate[0]
    assert snapshots[0].version > states[0].journal_version
    assert snapshots[1].state == states[1]
