from src.config import Config
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
