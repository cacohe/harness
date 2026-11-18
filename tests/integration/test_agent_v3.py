from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from datetime import datetime, timezone

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.application.composition import ApplicationConfig, create_development_agent
from src.core.artifacts import InlineContent
from src.core.commands import RunLimits
from src.core.context import (
    ContextBudget,
    ContextRequest,
    ContextWindow,
    EstimateQuality,
    TokenEstimate,
)
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.model import ModelMessage, ModelResponse, ModelRole, ModelToolCall
from src.core.state import RunStatus
from src.core.tools import ToolResult, ToolResultStatus, ToolSpec


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
        messages = (request.system_message, *request.open_tool_messages, request.current_input)
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

    async def complete(self, request):
        payload = json.loads(request.messages[-1].content)
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
        if round_number == 0 and message == "tools":
            return ModelResponse(
                tool_calls=(
                    ModelToolCall("call-first", "first", "{}"),
                    ModelToolCall("call-second", "second", "{}"),
                )
            )
        if round_number == 0 and message in {"approval", "reject", "cancel", "restart"}:
            return ModelResponse(tool_calls=(ModelToolCall("call-guarded", "guarded", "{}"),))
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


async def rig(*, journal=None, model=None, tools=None, ids=None):
    selected_journal = journal or InMemoryRunJournal()
    selected_model = model or Model()
    selected_tools = tools or Tools()
    repository = InMemoryArtifactRepository(1_048_576)
    externalizer = PolicyArtifactExternalizer(
        repository,
        ArtifactPolicy(
            64,
            1_048_576,
            256,
            frozenset({"application/json", "text/plain"}),
        ),
    )
    agent = await create_development_agent(
        ApplicationConfig(
            max_sessions=128,
            max_waiters_per_session=8,
            limits=RunLimits(16, 30.0, 64),
            context_budget=ContextBudget(512, 32, 16),
        ),
        model=selected_model,
        tools=selected_tools,
        context=Context(),
        context_request_factory=request_factory,
        ids=ids or Ids(),
        clock=Clock(),
        profile_id=ProfileId("profile"),
        journal=selected_journal,
        artifacts=externalizer,
    )
    return agent, selected_journal, selected_model, selected_tools


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
