from src.config import Config
from src.demo.agent import Agent
from src.schema.llm import FinishReason, LLMOutput
from src.schema.task import TaskPlan, TaskState


class Gateway:
    def __init__(self):
        self.model = "test-model"
        self.planning_calls = []

    def planning(self, context, stream=True, response_format=None):
        self.planning_calls.append((stream, response_format))
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlan(goal=["buy a phone"]),
        )


def test_agent_wires_local_tools_and_plans_a_new_task():
    gateway = Gateway()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    task = agent.run("buy a phone")

    assert gateway.planning_calls == [(False, TaskPlan)]
    assert task.state is TaskState.TASK_RUNNING
    assert task.plan.goal == ["buy a phone"]
    assert agent.to_dict()["model"] == "test-model"
    assert "shell" in agent.to_dict()["tools"]
    assert agent.harness._max_steps == 3
