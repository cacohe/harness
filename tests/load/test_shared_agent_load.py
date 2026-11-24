import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone

from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.context.tokenizers import ExactTokenCounter
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.composition import ApplicationConfig, create_development_agent
from src.core.context.contracts import (
    ContextBudget,
    ContextRequest,
    ContextWindow,
    EstimateQuality,
    TokenEstimate,
)
from src.core.context.model import ModelMessage, ModelResponse, ModelRole
from src.core.execution.commands import RunLimits
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.persistence.journal import ConversationSummaryCheckpoint


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
            context_counter=ExactTokenCounter(),
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
