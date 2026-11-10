import pytest
from src.config import Config
from src.core.llm import LlmCallError
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


class ChatGateway:
    def __init__(self, answers):
        self.model = "test-model"
        self.answers = list(answers)
        self.plans = []

    def planning(self, context, stream=True, response_format=None):
        self.plans.append(context.messages[1].content)
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlan(goal=["answer"]),
        )

    def infer(self, context, stream=True, response_format=None):
        return LLMOutput(FinishReason.STOP, self.answers.pop(0), [], None, None)


def test_chat_plans_then_returns_the_answer():
    gateway = ChatGateway(["hello"])
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    task = agent.chat("hi")

    assert task.state is TaskState.TASK_COMPLETED
    assert task.result.output == "hello"
    assert gateway.plans == ["hi"]
    assert agent.transcript == [("user", "hi"), ("assistant", "hello")]


def test_chat_sends_prior_turns_with_the_next_message():
    gateway = ChatGateway(["hello", "fine"])
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    agent.chat("hi")
    task = agent.chat("and you")

    assert task.result.output == "fine"
    assert gateway.plans[1] == "user: hi\nassistant: hello\nuser: and you"


def test_chat_drops_a_turn_that_fails():
    class FailingGateway(ChatGateway):
        def infer(self, context, stream=True, response_format=None):
            raise LlmCallError("down")

    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=FailingGateway([]),
    )

    with pytest.raises(LlmCallError, match="down"):
        agent.chat("hi")

    assert agent.transcript == []
