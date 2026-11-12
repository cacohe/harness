from src.config import Config
from src.demo.agent import Agent
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall
from src.schema.task import TaskPlan, TaskState


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


class FalseyConfig:
    llm = "test-model"
    llm_api_key = None
    llm_api_base = None
    max_steps = 3
    max_seconds = 30

    def __bool__(self):
        return False


class FalseyGateway(ChatGateway):
    def __bool__(self):
        return False


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


def test_failed_resume_clears_the_waiting_task():
    class FailingAfterPause(ChatGateway):
        def infer(self, context, stream=True, response_format=None):
            if not self.answers:
                raise ValueError("bad request")
            return self.answers.pop(0)

    gateway = FailingAfterPause(
        [
            LLMOutput(
                FinishReason.TOOL_CALLS,
                None,
                [LlmToolCall("call_1", "write_file", '{"path": "x.txt", "content": "x"}')],
                None,
                None,
            )
        ]
    )
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    assert agent.chat("write").state is TaskState.TASK_PAUSED

    [request] = agent.approval_requests()
    agent.decide_tool_call(request.tool_id, approved=False)
    task = agent.resume()

    assert task.state is TaskState.TASK_FAILED
    assert agent.pending_task is None
    assert agent.approval_requests() == []


def test_agent_preserves_explicit_falsey_dependencies(monkeypatch):
    monkeypatch.setattr(Config, "load", lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    config = FalseyConfig()
    gateway = FalseyGateway(["hello"])

    agent = Agent(config=config, llm_gateway=gateway)

    assert agent.config is config
    assert agent.llm_gateway is gateway


def test_from_config_builds_the_default_composition():
    config = Config(llm="test-model", max_steps=3, max_seconds=30)
    gateway = ChatGateway(["hello"])

    agent = Agent.from_config(config, llm_gateway=gateway)

    assert agent.config is config
    assert agent.llm_gateway is gateway
