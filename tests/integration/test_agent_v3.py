from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.tokenizers import ExactTokenCounter
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.composition import ApplicationConfig, create_development_agent
from src.core.artifacts.contracts import InlineContent
from src.core.context.contracts import (
    ContextBudget,
    ContextRequest,
    ContextWindow,
    EstimateQuality,
    TokenEstimate,
)
from src.core.context.model import ModelMessage, ModelResponse, ModelRole, ModelToolCall
from src.core.context.skills import (
    ResolvedSkill,
    SkillBinding,
    SkillCatalogEntry,
    skill_content_sha256,
)
from src.core.execution.commands import RunLimits
from src.core.execution.errors import RunErrorCode, RunFailure
from src.core.execution.projection import LOAD_SKILL_TOOL_NAME
from src.core.execution.state import RunStatus
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.tools.contracts import ToolResult, ToolResultStatus, ToolSpec


class Ids:
    def __init__(self, namespace: str = "ids") -> None:
        self._next_value = 0
        self._namespace = namespace

    def _next(self, prefix: str) -> str:
        self._next_value += 1
        return f"{self._namespace}-{prefix}-{self._next_value}"

    def run(self) -> RunId:
        return RunId(self._next("run"))

    def session(self) -> SessionId:
        return SessionId(self._next("session"))

    def submission(self) -> SubmissionId:
        return SubmissionId(self._next("submission"))

    def profile(self) -> ProfileId:
        return ProfileId(self._next("profile"))

    def event(self) -> EventId:
        return EventId(self._next("event"))

    def step(self) -> str:
        return self._next("step")

    def model_call(self) -> str:
        return self._next("model")


class Clock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


TOOLS = (
    ToolSpec("first", "first deterministic tool", "{}"),
    ToolSpec("second", "second deterministic tool", "{}"),
    ToolSpec(
        "guarded",
        "approval-bound deterministic tool",
        "{}",
        requires_approval=True,
        approval_scope="write",
        workspace_id="workspace",
    ),
    ToolSpec("slow", "stable timeout result", "{}"),
)


class Context:
    async def build(self, request: ContextRequest, budget: ContextBudget) -> ContextWindow:
        messages = (
            request.system_message,
            *request.skill_messages,
            request.current_input,
            *request.open_tool_messages,
        )
        return ContextWindow(
            messages,
            request.tools,
            TokenEstimate(len(messages), EstimateQuality.EXACT),
            budget,
            (),
            request.policy_version,
        )


class Model:
    def __init__(self, concurrent_calls: int = 0) -> None:
        self.calls: Counter[tuple[str, str]] = Counter()
        self._expected_concurrent = concurrent_calls
        self._arrived = 0
        self._all_arrived = asyncio.Event()
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        payload = json.loads(
            next(
                message.content
                for message in reversed(request.messages)
                if message.role is ModelRole.USER
            )
        )
        session = payload["session"]
        message = payload["message"]
        round_number = payload["round"]
        self.calls[(session, message)] += 1
        if message.startswith("concurrent-"):
            self._arrived += 1
            if self._arrived == self._expected_concurrent:
                self._all_arrived.set()
            await self._all_arrived.wait()
        if message == "model-timeout":
            raise TimeoutError("provider deadline")
        if message == "artifact-lifecycle":
            return ModelResponse("x" * 80)
        if round_number == 0 and message == "skills":
            return ModelResponse(
                tool_calls=(
                    ModelToolCall(
                        "call-load-skill",
                        LOAD_SKILL_TOOL_NAME,
                        '{"name":"review"}',
                    ),
                )
            )
        if round_number == 0 and message == "final-step-tools":
            return ModelResponse(
                tool_calls=(
                    ModelToolCall("call-first", "first", "{}"),
                    ModelToolCall("call-guarded", "guarded", "{}"),
                )
            )
        if round_number == 0 and message == "tools":
            return ModelResponse(
                tool_calls=(
                    ModelToolCall("call-first", "first", "{}"),
                    ModelToolCall("call-second", "second", "{}"),
                )
            )
        if message == "two-tool-batches" and round_number < 2:
            name = "first" if round_number == 0 else "second"
            return ModelResponse(tool_calls=(ModelToolCall(f"call-{name}", name, "{}"),))
        if round_number == 0 and message in {"approval", "reject", "cancel", "restart"}:
            return ModelResponse(tool_calls=(ModelToolCall("call-guarded", "guarded", "{}"),))
        if round_number == 0 and message == "approval-batch":
            return ModelResponse(
                tool_calls=(
                    ModelToolCall("call-guarded-first", "guarded", "{}"),
                    ModelToolCall("call-guarded-second", "guarded", "{}"),
                )
            )
        if round_number == 0 and message == "tool-timeout":
            return ModelResponse(tool_calls=(ModelToolCall("call-slow", "slow", "{}"),))
        return ModelResponse(f"answer:{message}")


class Tools:
    def __init__(self) -> None:
        self.effects: Counter[str] = Counter()
        self._results: dict[str, ToolResult] = {}

    async def execute(self, request):
        cached = self._results.get(request.idempotency_key)
        if cached is not None:
            return cached
        self.effects[request.tool_call_id] += 1
        if request.name == "slow":
            result = ToolResult(
                ToolResultStatus.TIMED_OUT,
                failure=RunFailure(RunErrorCode.TOOL_TIMED_OUT),
                execution_key=request.idempotency_key,
            )
        else:
            result = ToolResult(
                ToolResultStatus.SUCCESS,
                inline_content=InlineContent({"call": request.tool_call_id}, "application/json"),
                execution_key=request.idempotency_key,
            )
        self._results[request.idempotency_key] = result
        return result


class Skills:
    def __init__(self) -> None:
        message = ModelMessage(ModelRole.SYSTEM, "ACTIVE REVIEW INSTRUCTIONS")
        self.binding = SkillBinding("review", "v1", skill_content_sha256((message,)))
        self.calls = []

    async def catalog(self, profile_id):
        self.calls.append("catalog")
        return (SkillCatalogEntry("review", "Review changes", "v1"),)

    async def activate(self, name, *, profile_id):
        self.calls.append("activate")
        assert name == "review"
        return ResolvedSkill(
            self.binding,
            (ModelMessage(ModelRole.SYSTEM, "ACTIVE REVIEW INSTRUCTIONS"),),
        )

    async def resolve(self, binding, *, profile_id):
        self.calls.append("resolve")
        assert binding == self.binding
        return ResolvedSkill(
            self.binding,
            (ModelMessage(ModelRole.SYSTEM, "ACTIVE REVIEW INSTRUCTIONS"),),
        )


def request_factory(state, message: str) -> ContextRequest:
    payload = json.dumps(
        {
            "session": state.session_id.value,
            "message": message,
            "round": len(state.used_step_ids),
        },
        separators=(",", ":"),
    )
    return ContextRequest(
        state.session_id,
        state.profile_id,
        "integration-v1",
        ModelMessage(ModelRole.SYSTEM, "integration system"),
        ModelMessage(ModelRole.USER, payload),
        TOOLS,
    )


async def rig(
    *,
    journal=None,
    model=None,
    tools=None,
    ids=None,
    observer=None,
    context=None,
    limits=None,
    skills=None,
):
    selected_journal = journal or InMemoryRunJournal()
    selected_model = model or Model()
    selected_tools = tools or Tools()
    repository = InMemoryArtifactRepository(1_048_576)
    externalizer = PolicyArtifactExternalizer(
        repository,
        ArtifactPolicy(
            64,
            1_048_576,
            64,
            frozenset({"application/json", "text/plain"}),
        ),
    )
    agent = await create_development_agent(
        ApplicationConfig(
            max_sessions=128,
            max_waiters_per_session=8,
            limits=limits or RunLimits(16, 30.0, 64),
            context_budget=ContextBudget(2_048, 32, 16),
        ),
        model=selected_model,
        tools=selected_tools,
        context=context or Context(),
        context_counter=ExactTokenCounter(),
        context_request_factory=request_factory,
        ids=ids or Ids(),
        clock=Clock(),
        profile_id=ProfileId("profile"),
        journal=selected_journal,
        artifacts=externalizer,
        observer=observer,
        skills=skills,
    )
    return agent, selected_journal, selected_model, selected_tools


def test_progressive_skill_catalog_activation_and_persistent_context():
    async def scenario():
        skills = Skills()
        model = Model()
        tools = Tools()
        agent, _, _, _ = await rig(
            model=model,
            tools=tools,
            skills=skills,
            limits=RunLimits(16, 30.0, 256),
        )
        session = await agent.open_session(SessionId("skills"))
        state = await session.submit("skills", SubmissionId("skills"))
        await agent.close()
        return state, model, tools, skills

    state, model, tools, skills = asyncio.run(scenario())
    assert state.status is RunStatus.COMPLETED
    assert state.active_skills == (skills.binding,)
    first_system = "\n".join(
        message.content or ""
        for message in model.requests[0].messages
        if message.role is ModelRole.SYSTEM
    )
    assert '"description":"Review changes"' in first_system
    assert "ACTIVE REVIEW INSTRUCTIONS" not in first_system
    assert sum(tool.name == LOAD_SKILL_TOOL_NAME for tool in model.requests[0].tools) == 1
    assert any(
        message.content == "ACTIVE REVIEW INSTRUCTIONS" for message in model.requests[1].messages
    )
    assert skills.calls == ["catalog", "activate", "catalog", "resolve"]
    assert tools.effects == Counter()


def test_real_bounded_context_deduplicates_current_input_and_protects_latest_tool_batch():
    class CapturingBoundedContext(BoundedContextProvider):
        def __init__(self, journal):
            super().__init__(journal, ExactTokenCounter(), page_size=8)
            self.context_requests = []

        async def build(self, request, budget):
            self.context_requests.append(request)
            return await super().build(request, budget)

    async def scenario():
        journal = InMemoryRunJournal()
        context = CapturingBoundedContext(journal)
        model = Model()
        agent, _, _, _ = await rig(journal=journal, model=model, context=context)
        session = await agent.open_session(SessionId("real-context"))
        state = await session.submit("tools", SubmissionId("real-context"))
        await agent.close()
        return state, context, model

    state, context, model = asyncio.run(scenario())
    assert state.status is RunStatus.COMPLETED
    assert len(context.context_requests) == 2
    assert context.context_requests[1].open_tool_messages
    current = context.context_requests[1].current_input
    open_tools = context.context_requests[1].open_tool_messages
    assert model.requests[1].messages[-(len(open_tools) + 1) :] == (current, *open_tools)
    for request in model.requests:
        assert sum(message.role is ModelRole.USER for message in request.messages) == 1
    assert [message.role for message in context.context_requests[1].open_tool_messages] == [
        ModelRole.ASSISTANT,
        ModelRole.TOOL,
        ModelRole.TOOL,
    ]


def test_real_bounded_context_keeps_all_current_run_tool_batches_after_user_input():
    class CapturingBoundedContext(BoundedContextProvider):
        def __init__(self, journal):
            super().__init__(journal, ExactTokenCounter(), page_size=8)

    async def scenario():
        journal = InMemoryRunJournal()
        model = Model()
        agent, _, _, _ = await rig(
            journal=journal,
            model=model,
            context=CapturingBoundedContext(journal),
        )
        session = await agent.open_session(SessionId("multi-batch-context"))
        state = await session.submit("two-tool-batches", SubmissionId("multi-batch-context"))
        await agent.close()
        return state, model.requests

    state, requests = asyncio.run(scenario())
    assert state.status is RunStatus.COMPLETED
    assert len(requests) == 3
    assert [message.role for message in requests[2].messages] == [
        ModelRole.SYSTEM,
        ModelRole.USER,
        ModelRole.ASSISTANT,
        ModelRole.TOOL,
        ModelRole.ASSISTANT,
        ModelRole.TOOL,
    ]


def test_cancelled_submit_is_recovered_before_the_next_submission():
    class InterruptOnceModel(Model):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()
            self._interrupt = True

        async def complete(self, request):
            if self._interrupt:
                self._interrupt = False
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            return await super().complete(request)

    async def scenario():
        model = InterruptOnceModel()
        agent, journal, _, _ = await rig(model=model)
        session = await agent.open_session(SessionId("recover-cancelled"))
        interrupted = asyncio.create_task(
            session.submit("interrupted", SubmissionId("interrupted"))
        )
        await model.entered.wait()
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted
        assert model.cancelled.is_set()
        current = await session.current_run()
        completed = await session.submit("next", SubmissionId("next"))
        recovered = await journal.find_run_state(
            SessionId("recover-cancelled"), current.run_id, current.submission_id
        )
        await agent.close()
        return current, recovered, completed

    current, recovered, completed = asyncio.run(scenario())
    assert current.status is RunStatus.RUNNING
    assert recovered is not None and recovered.status is RunStatus.FAILED
    assert completed.status is RunStatus.COMPLETED


def test_agent_runs_direct_answer_and_ordered_tool_batch_end_to_end():
    async def scenario():
        agent, journal, _, tools = await rig()
        direct = await agent.submit("direct", SubmissionId("direct"))
        tool_session = await agent.open_session(SessionId("tool-session"))
        tool_run = await tool_session.submit("tools", SubmissionId("tools"))
        snapshot = await journal.load_session(SessionId("tool-session"))
        await agent.close()
        return direct, tool_run, snapshot, tools

    direct, tool_run, snapshot, tools = asyncio.run(scenario())
    assert direct.status is RunStatus.COMPLETED
    assert direct.final_text == "answer:direct"
    assert tool_run.status is RunStatus.COMPLETED
    assert tool_run.final_text == "answer:tools"
    assert tools.effects == {"call-first": 1, "call-second": 1}
    assert [entry.role for entry in snapshot.transcript] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]


def test_agent_final_step_tool_batch_has_no_tool_or_approval_side_effect() -> None:
    async def scenario():
        agent, journal, _, tools = await rig(limits=RunLimits(1, 30.0, 64))
        session = await agent.open_session(SessionId("final-step-tools"))
        state = await session.submit("final-step-tools", SubmissionId("final-step-tools"))
        snapshot = await journal.load_session(SessionId("final-step-tools"))
        await agent.close()
        return state, snapshot, tools

    state, snapshot, tools = asyncio.run(scenario())
    assert state.status is RunStatus.FAILED
    assert state.failure == RunFailure(RunErrorCode.LIMIT_EXCEEDED)
    assert tools.effects == {}
    assert [entry.role for entry in snapshot.transcript] == ["user"]


def test_agent_session_publicly_detaches_production_generated_artifacts() -> None:
    async def scenario():
        agent, journal, _, _ = await rig()
        session = await agent.open_session(SessionId("artifact-lifecycle"))
        state = await session.submit("artifact-lifecycle", SubmissionId("artifact-lifecycle"))
        assert state.status is RunStatus.COMPLETED
        assert state.final_artifacts
        active = await journal.load_artifact_attachments(state.session_id, state.run_id)
        detached = await session.detach_artifacts(state.run_id, roles=(active[0].role,))
        repeated = await session.detach_artifacts(state.run_id)
        facts = await journal.load_artifact_facts(state.session_id, state.run_id, -1, 16)
        await agent.close()
        return active, detached, repeated, facts

    active, detached, repeated, facts = asyncio.run(scenario())
    assert len(active) == 1
    assert detached[0].artifact == active[0].artifact
    assert detached[0].owner_id == active[0].owner_id
    assert detached[0].action.value == "detached"
    assert repeated == ()
    assert [record.fact.action.value for record in facts] == ["attached", "detached"]


def test_detach_commit_survives_observer_failure_and_retry() -> None:
    class Observer:
        async def observe(self, observation):
            if observation.kind == "artifact.detached":
                raise RuntimeError("observer unavailable")

    async def scenario():
        agent, journal, _, _ = await rig(observer=Observer())
        session = await agent.open_session(SessionId("artifact-observer"))
        state = await session.submit("artifact-lifecycle", SubmissionId("artifact-observer"))
        first = await session.detach_artifacts(state.run_id)
        retry = await session.detach_artifacts(state.run_id)
        active = await journal.load_artifact_attachments(state.session_id, state.run_id)
        await agent.close()
        return first, retry, active

    first, retry, active = asyncio.run(scenario())
    assert len(first) == 1
    assert retry == ()
    assert active == ()


@pytest.mark.parametrize("approved", [True, False])
def test_approval_decision_resumes_or_rejects_without_replaying_side_effect(approved):
    async def scenario():
        agent, _, _, tools = await rig()
        session = await agent.open_session(SessionId(f"approval-{approved}"))
        waiting = await session.submit(
            "approval" if approved else "reject", SubmissionId(f"approval-{approved}")
        )
        settled = await session.decide_tool_call(waiting.run_id, "call-guarded", approved)
        await agent.close()
        return waiting, settled, tools

    waiting, settled, tools = asyncio.run(scenario())
    assert waiting.status is RunStatus.WAITING_APPROVAL
    assert settled.status is RunStatus.COMPLETED
    assert settled.final_text == f"answer:{'approval' if approved else 'reject'}"
    assert tools.effects["call-guarded"] == (1 if approved else 0)


def test_cancel_waiting_run_is_durable_and_never_executes_tool():
    async def scenario():
        agent, journal, _, tools = await rig()
        session = await agent.open_session(SessionId("cancel-session"))
        waiting = await session.submit("cancel", SubmissionId("cancel"))
        cancelled = await session.cancel(waiting.run_id)
        head = await journal.load_head(SessionId("cancel-session"))
        await agent.close()
        return cancelled, head, tools

    cancelled, head, tools = asyncio.run(scenario())
    assert cancelled.status is RunStatus.CANCELLED
    assert head.state == cancelled
    assert tools.effects["call-guarded"] == 0


@pytest.mark.parametrize("blocked_adapter", ["model", "tool"])
def test_cancel_interrupts_blocked_model_or_tool_adapter(blocked_adapter):
    class BlockingModel(Model):
        entered = asyncio.Event()
        interrupted = asyncio.Event()

        async def complete(self, request):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.interrupted.set()
                raise

    class BlockingTools(Tools):
        entered = asyncio.Event()
        interrupted = asyncio.Event()

        async def execute(self, request):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.interrupted.set()
                raise

    async def scenario():
        model = BlockingModel() if blocked_adapter == "model" else Model()
        tools = BlockingTools() if blocked_adapter == "tool" else Tools()
        agent, journal, _, _ = await rig(model=model, tools=tools)
        session_id = SessionId(f"blocked-{blocked_adapter}")
        session = await agent.open_session(session_id)
        submit = asyncio.create_task(
            session.submit(
                "direct" if blocked_adapter == "model" else "tools",
                SubmissionId(f"blocked-{blocked_adapter}"),
            )
        )
        blocker = model if blocked_adapter == "model" else tools
        await blocker.entered.wait()
        head = await journal.load_head(session_id)
        assert head.state is not None
        cancelled = await asyncio.wait_for(session.cancel(head.state.run_id), 1)
        with pytest.raises(asyncio.CancelledError):
            await submit
        await agent.close()
        return cancelled, blocker

    cancelled, blocker = asyncio.run(scenario())
    assert cancelled.status is RunStatus.CANCELLED
    assert blocker.interrupted.is_set()


def test_model_and_tool_timeout_are_stably_classified():
    async def scenario():
        agent, journal, _, _ = await rig()
        model_session = await agent.open_session(SessionId("model-timeout"))
        model_result = await model_session.submit("model-timeout", SubmissionId("model-timeout"))
        tool_session = await agent.open_session(SessionId("tool-timeout"))
        tool_result = await tool_session.submit("tool-timeout", SubmissionId("tool-timeout"))
        snapshot = await journal.load_session(SessionId("tool-timeout"))
        await agent.close()
        return model_result, tool_result, snapshot

    model_result, tool_result, snapshot = asyncio.run(scenario())
    assert model_result.failure == RunFailure(RunErrorCode.MODEL_FAILED)
    assert tool_result.status is RunStatus.COMPLETED
    assert any("tool_timed_out" in entry.content for entry in snapshot.transcript)


def test_restart_recovers_waiting_approval_and_submission_replay_is_idempotent():
    async def scenario():
        journal = InMemoryRunJournal()
        tools = Tools()
        first_agent, _, first_model, _ = await rig(
            journal=journal, tools=tools, ids=Ids("before-restart")
        )
        session = await first_agent.open_session(SessionId("restart-session"))
        waiting = await session.submit("restart", SubmissionId("restart-submission"))
        await first_agent.close()

        second_agent, _, second_model, _ = await rig(
            journal=journal, tools=tools, ids=Ids("after-restart")
        )
        restored = await second_agent.open_session(SessionId("restart-session"))
        completed = await restored.decide_tool_call(waiting.run_id, "call-guarded", True)
        replay = await restored.submit("restart", SubmissionId("restart-submission"))
        await second_agent.close()
        return waiting, completed, replay, first_model, second_model, tools

    waiting, completed, replay, first_model, second_model, tools = asyncio.run(scenario())
    assert waiting.status is RunStatus.WAITING_APPROVAL
    assert completed.status is RunStatus.COMPLETED
    assert replay == completed
    assert tools.effects["call-guarded"] == 1
    assert sum(first_model.calls.values()) == 1
    assert sum(second_model.calls.values()) == 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_historical_submission_replay_returns_original_terminal_outcome_after_restart(
    backend, tmp_path
):
    async def scenario():
        path = (tmp_path / "historical.sqlite3").resolve()
        first_journal = InMemoryRunJournal() if backend == "memory" else SQLiteRunJournal(path)
        model = Model()
        first_agent, _, _, _ = await rig(journal=first_journal, model=model)
        session = await first_agent.open_session(SessionId("historical-session"))
        first = await session.submit("first", SubmissionId("first"))
        second = await session.submit("second", SubmissionId("second"))
        head_before = await first_journal.load_head(SessionId("historical-session"))
        await first_agent.close()

        restarted_journal = first_journal if backend == "memory" else SQLiteRunJournal(path)
        second_agent, _, _, _ = await rig(journal=restarted_journal, model=model)
        restored = await second_agent.open_session(SessionId("historical-session"))
        replay = await restored.submit("first", SubmissionId("first"))
        head_after = await restarted_journal.load_head(SessionId("historical-session"))
        await second_agent.close()
        return first, second, replay, head_before, head_after, model

    first, second, replay, head_before, head_after, model = asyncio.run(scenario())
    assert first.final_text == "answer:first"
    assert second.final_text == "answer:second"
    assert replay == first
    assert head_after == head_before
    assert sum(model.calls.values()) == 2


def test_restart_resumes_durable_decision_in_multi_tool_batch_without_repeating_effects():
    class CrashAfterDecision:
        armed = False

        async def observe(self, observation):
            if self.armed and observation.status == RunStatus.WAITING_APPROVAL.value:
                self.armed = False
                raise asyncio.CancelledError

    async def scenario():
        journal = InMemoryRunJournal()
        tools = Tools()
        observer = CrashAfterDecision()
        first_agent, _, _, _ = await rig(
            journal=journal, tools=tools, ids=Ids("batch-before"), observer=observer
        )
        first = await first_agent.open_session(SessionId("approval-batch-session"))
        waiting = await first.submit("approval-batch", SubmissionId("approval-batch"))
        observer.armed = True
        with pytest.raises(asyncio.CancelledError):
            await first.decide_tool_call(waiting.run_id, "call-guarded-first", True)
        decided_head = await journal.load_head(SessionId("approval-batch-session"))
        await first_agent.close()

        second_agent, _, _, _ = await rig(journal=journal, tools=tools, ids=Ids("batch-after"))
        restored = await second_agent.open_session(SessionId("approval-batch-session"))
        next_waiting = await restored.submit("approval-batch", SubmissionId("approval-batch"))
        completed = await restored.decide_tool_call(waiting.run_id, "call-guarded-second", False)
        replay = await restored.submit("approval-batch", SubmissionId("approval-batch"))
        await second_agent.close()
        return decided_head, next_waiting, completed, replay, tools

    decided_head, next_waiting, completed, replay, tools = asyncio.run(scenario())
    assert decided_head.state is not None
    assert decided_head.state.status is RunStatus.WAITING_APPROVAL
    assert next_waiting.status is RunStatus.WAITING_APPROVAL
    assert completed.status is RunStatus.COMPLETED
    assert replay == completed
    assert tools.effects["call-guarded-first"] == 1
    assert tools.effects["call-guarded-second"] == 0


def test_different_sessions_execute_concurrently_without_state_leakage():
    async def scenario():
        model = Model(concurrent_calls=2)
        agent, _, _, _ = await rig(model=model)
        sessions = [
            await agent.open_session(SessionId("concurrent-a")),
            await agent.open_session(SessionId("concurrent-b")),
        ]
        states = await asyncio.wait_for(
            asyncio.gather(
                sessions[0].submit("concurrent-a", SubmissionId("concurrent-a")),
                sessions[1].submit("concurrent-b", SubmissionId("concurrent-b")),
            ),
            timeout=2,
        )
        await agent.close()
        return states, model

    states, model = asyncio.run(scenario())
    assert [state.final_text for state in states] == [
        "answer:concurrent-a",
        "answer:concurrent-b",
    ]
    assert model.calls == {
        ("concurrent-a", "concurrent-a"): 1,
        ("concurrent-b", "concurrent-b"): 1,
    }
