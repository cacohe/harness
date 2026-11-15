import asyncio
import json

from src.config import Config
from src.core.memory import MemoryManager
from src.demo.agent import Agent
from src.memory.retriever import HybridRetriever
from src.memory.store import InMemoryStore
from src.observability import MetricsEventSink
from src.persistence.in_memory import InMemoryTaskRepository
from src.schema.llm import FinishReason, LLMOutput
from src.schema.task import TaskPlan, TaskState


class YieldingGateway:
    """Deterministic async gateway that forces sessions to interleave."""

    def __init__(self, expected_calls=64):
        self.expected_calls = expected_calls
        self.planning_inputs = []
        self.infer_inputs = []
        self.planning_arrived = 0
        self.all_planning_arrived = asyncio.Event()

    @staticmethod
    def _current_message(context):
        content = context.messages[1].content or ""
        marker = "Current user message:\n"
        return content.split(marker, 1)[1] if marker in content else content

    async def aplanning(self, context, stream=True, response_format=None):
        del stream, response_format
        current = self._current_message(context)
        self.planning_inputs.append(current)
        self.planning_arrived += 1
        if self.planning_arrived == self.expected_calls:
            self.all_planning_arrived.set()
        await self.all_planning_arrived.wait()
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlan(goal=["answer"]),
        )

    async def ainfer(self, context, stream=True, response_format=None):
        del stream, response_format
        await asyncio.sleep(0)
        current = self._current_message(context)
        self.infer_inputs.append(current)
        return LLMOutput(FinishReason.STOP, f"answer:{current}", [], None, None)


def test_shared_agent_handles_concurrent_sessions_without_state_leaks():
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
    assert set(gateway.planning_inputs) == expected_inputs
    assert set(gateway.infer_inputs) == expected_inputs
    assert gateway.planning_arrived == 64
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
    assert snapshot["events"]["llm_started"] == 128
    assert snapshot["events"]["llm_completed"] == 128
    assert snapshot["task_outcomes"] == {"completed": 64}
    assert snapshot["in_flight"] == {"tasks": 0, "llm": 0, "tools": 0}
    assert snapshot["inflight_evictions"] == {"tasks": 0, "llm": 0, "tools": 0}
    assert snapshot["durations"]["task"]["count"] == 64
    assert snapshot["durations"]["llm"]["count"] == 128
