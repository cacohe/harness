import asyncio
import re

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

    async def aplanning(self, context, stream=True, response_format=None):
        del context, stream, response_format
        await asyncio.sleep(0)
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
        identifiers = re.findall(
            r"load-\d+", "\n".join(message.content or "" for message in context.messages)
        )
        assert identifiers
        return LLMOutput(FinishReason.STOP, f"answer:{identifiers[-1]}", [], None, None)


def test_shared_agent_handles_concurrent_sessions_without_state_leaks():
    async def scenario():
        repository = InMemoryTaskRepository()
        memory = MemoryManager(InMemoryStore(), HybridRetriever())
        metrics = MetricsEventSink(max_inflight=1_000)
        agent = Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=YieldingGateway(),
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
        return repository, memory, metrics, sessions, tasks

    repository, memory, metrics, sessions, tasks = asyncio.run(scenario())

    assert len({task.id for task in tasks}) == 64
    assert all(task.state is TaskState.TASK_COMPLETED for task in tasks)
    for index, (session, task) in enumerate(zip(sessions, tasks, strict=True)):
        expected = f"answer:load-{index}"
        assert task.result.output == expected
        assert session.transcript == [("user", f"load-{index}"), ("assistant", expected)]
        assert repository.get(task.id) == task

    assert len(memory.recent(64)) == 64
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
