import asyncio

import pytest
from src.config import Config
from src.core.cancellation import CancellationToken
from src.core.loop import Harness
from src.core.task import Task
from src.core.workspace import Workspace
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


class FalseyWorkspace(Workspace):
    def __bool__(self):
        return False


class FalseySandbox:
    def __bool__(self):
        return False

    def supports(self, tool):
        return True

    def run(self, tool, inputs, workspace):
        return "sandboxed"


class FalseyCancellationToken(CancellationToken):
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


def test_chat_records_missing_model_content_as_an_empty_string():
    gateway = ChatGateway([None, "next"])
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    first = agent.chat("empty")
    agent.chat("continue")

    assert first.result.output == ""
    assert agent.transcript[:2] == [("user", "empty"), ("assistant", "")]
    assert gateway.plans[1] == "user: empty\nassistant: \nuser: continue"


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


def test_default_agent_registers_workspace_file_tools_without_shell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
    )

    assert {tool.name for tool in agent.tool_registry.get_all_tools()} == {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "glob",
    }


def test_default_agent_binds_file_tools_to_explicit_workspace(tmp_path):
    workspace = Workspace(tmp_path)

    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
        workspace=workspace,
    )

    tools = agent.tool_registry.get_all_tools()
    assert agent.workspace is workspace
    assert tools
    assert all(tool.workspace is workspace for tool in tools)


def test_default_agent_does_not_advertise_shell_without_sandbox(tmp_path):
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
        workspace=tmp_path,
    )

    assert "shell" not in {tool.name for tool in agent.tool_registry.get_all_tools()}


def test_default_agent_registers_shell_and_passes_workspace_to_sandbox(tmp_path):
    workspace = Workspace(tmp_path)
    sandbox = FalseySandbox()

    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
        workspace=workspace,
        sandbox_runner=sandbox,
    )

    shell = agent.tool_registry.find_tool("shell")
    assert shell is not None
    assert shell.workspace is workspace
    assert agent.tool_executor.sandbox_runner is sandbox


def test_agent_preserves_falsey_workspace_and_sandbox_dependencies(tmp_path):
    workspace = FalseyWorkspace(tmp_path)
    sandbox = FalseySandbox()

    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
        workspace=workspace,
        sandbox_runner=sandbox,
    )

    assert agent.workspace is workspace
    assert agent.tool_executor.sandbox_runner is sandbox
    assert agent.tool_registry.find_tool("shell") is not None


def test_chat_passes_the_exact_falsey_cancellation_token(monkeypatch):
    observed = []

    async def record_run(_harness, task, cancellation_token=None):
        observed.append(cancellation_token)
        return task.result

    monkeypatch.setattr(Harness, "arun", record_run)
    token = FalseyCancellationToken()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
    )

    agent.chat("cancel me", cancellation_token=token)

    assert observed == [token]


def test_resume_passes_the_exact_falsey_cancellation_token(monkeypatch):
    observed = []

    async def record_run(_harness, task, cancellation_token=None):
        observed.append(cancellation_token)
        return task.result

    monkeypatch.setattr(Harness, "arun", record_run)
    token = FalseyCancellationToken()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
    )
    agent.pending_task = Task("pending")

    agent.resume(cancellation_token=token)

    assert observed == [token]


def test_cancelled_agent_turn_is_not_added_to_transcript():
    token = CancellationToken()
    token.cancel()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["unused"]),
    )

    task = agent.chat("cancel me", cancellation_token=token)

    assert task.state is TaskState.TASK_FAILED
    assert agent.transcript == []


def test_achat_completes_and_remembers_the_turn():
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
    )

    task = asyncio.run(agent.achat("hi"))

    assert task.result.output == "hello"
    assert agent.transcript == [("user", "hi"), ("assistant", "hello")]


def test_achat_pauses_and_aresume_finishes_after_a_decision():
    class ApprovalGateway(ChatGateway):
        def infer(self, context, stream=True, response_format=None):
            return self.answers.pop(0)

    gateway = ApprovalGateway(
        [
            LLMOutput(
                FinishReason.TOOL_CALLS,
                None,
                [LlmToolCall("call-1", "write_file", '{"path":"x.txt","content":"x"}')],
                None,
                None,
            ),
            LLMOutput(FinishReason.STOP, "done", [], None, None),
        ]
    )
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    paused = asyncio.run(agent.achat("write"))
    assert paused.state is TaskState.TASK_PAUSED
    agent.decide_tool_call("call-1", approved=False)
    completed = asyncio.run(agent.aresume())

    assert completed.result.output == "done"
    assert agent.pending_task is None
    assert agent.transcript == [("user", "write"), ("assistant", "done")]


def test_achat_passes_the_exact_falsey_cancellation_token(monkeypatch):
    observed = []

    async def record_run(_harness, task, cancellation_token=None):
        observed.append(cancellation_token)
        return task.result

    monkeypatch.setattr(Harness, "arun", record_run)
    token = FalseyCancellationToken()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
    )

    asyncio.run(agent.achat("cancel me", cancellation_token=token))

    assert observed == [token]


def test_failed_achat_is_not_added_to_transcript():
    token = CancellationToken()
    token.cancel()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["unused"]),
    )

    task = asyncio.run(agent.achat("cancel me", cancellation_token=token))

    assert task.state is TaskState.TASK_FAILED
    assert agent.transcript == []


def test_independent_agents_can_chat_concurrently():
    class RendezvousGateway:
        def __init__(self, rendezvous):
            self.rendezvous = rendezvous

        async def aplanning(self, context, stream=True, response_format=None):
            return LLMOutput(
                FinishReason.STOP,
                None,
                [],
                None,
                None,
                structured=TaskPlan(goal=["answer"]),
            )

        async def ainfer(self, context, stream=True, response_format=None):
            self.rendezvous["count"] += 1
            if self.rendezvous["count"] == 2:
                self.rendezvous["ready"].set()
            await self.rendezvous["ready"].wait()
            return LLMOutput(FinishReason.STOP, "done", [], None, None)

    async def scenario():
        rendezvous = {"count": 0, "ready": asyncio.Event()}
        config = Config(llm="test-model", max_steps=3, max_seconds=30)
        first = Agent(config=config, llm_gateway=RendezvousGateway(rendezvous))
        second = Agent(config=config, llm_gateway=RendezvousGateway(rendezvous))
        return await asyncio.wait_for(
            asyncio.gather(first.achat("one"), second.achat("two")), timeout=1
        )

    tasks = asyncio.run(scenario())

    assert [task.result.output for task in tasks] == ["done", "done"]


def test_chat_rejects_an_active_event_loop_before_starting_work():
    gateway = ChatGateway(["unused"])
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
    )

    async def scenario():
        with pytest.raises(RuntimeError, match="await achat"):
            agent.chat("hi")

    asyncio.run(scenario())

    assert gateway.plans == []
