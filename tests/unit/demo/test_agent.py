import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from src.config import Config
from src.core.cancellation import CancellationToken
from src.core.loop import Harness
from src.core.task import Task
from src.core.workspace import Workspace
from src.demo.agent import Agent
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall
from src.schema.task import TaskPlan, TaskState
from src.schema.tool import ApprovalDecision, Tool
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry


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


class RecordingSandbox(FalseySandbox):
    def __init__(self):
        self.calls = []

    def run(self, tool, inputs, workspace):
        self.calls.append((tool, inputs, workspace))
        return "sandboxed"


class FalseyCancellationToken(CancellationToken):
    def __bool__(self):
        return False


class RecordingMCPManager:
    def __init__(self, catalog):
        self.catalog = catalog
        self.calls = []
        self.close_calls = 0

    async def add_server(self, name, source, **options):
        self.calls.append(("add", name, source, options))

    async def refresh_server(self, name):
        self.calls.append(("refresh", name))

    async def remove_server(self, name):
        self.calls.append(("remove", name))

    async def aclose(self):
        self.close_calls += 1


def _write_skill(root: Path, folder: str, name: str) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Helps with research.\n---\nFollow the evidence.\n",
        encoding="utf-8",
    )
    (skill_dir / "reference.md").write_text("trusted", encoding="utf-8")
    return skill_dir


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


def test_default_composition_shares_one_new_catalog():
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
    )

    assert isinstance(agent.tool_registry, DefaultToolRegistry)
    assert agent.tool_executor.catalog is agent.tool_registry
    assert agent.mcp_manager.catalog is agent.tool_registry


def test_agent_loads_skill_into_prompt_and_registers_read_tool_without_shell(tmp_path):
    class PromptGateway(ChatGateway):
        def __init__(self):
            super().__init__(["done"])
            self.system_prompts = []

        def infer(self, context, stream=True, response_format=None):
            self.system_prompts.append(context.messages[0].content)
            return super().infer(context, stream=stream, response_format=response_format)

    skill_dir = _write_skill(tmp_path, "research", "research helper")
    gateway = PromptGateway()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=gateway,
        workspace=tmp_path,
        skill_directories=[skill_dir],
    )

    agent.chat("research this")

    assert [skill.name for skill in agent.harness.skills] == ["research helper"]
    assert "Follow the evidence." in gateway.system_prompts[0]
    assert "skill__research_helper__read_file" in gateway.system_prompts[0]
    assert agent.tool_registry.find_tool("skill__research_helper__read_file") is not None
    assert agent.tool_registry.find_tool("skill__research_helper__shell") is None


def test_agent_routes_skill_shell_through_sandbox_with_skill_workspace(tmp_path):
    skill_dir = _write_skill(tmp_path, "scripts", "script runner")
    sandbox = RecordingSandbox()
    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["unused"]),
        workspace=tmp_path,
        sandbox_runner=sandbox,
        skill_directories=[skill_dir],
    )

    held = agent.tool_executor.execute_tool_call(
        LlmToolCall(
            "call-1",
            "skill__script_runner__shell",
            '{"command":"bash scripts/check.sh"}',
        )
    )

    assert held.is_pending()
    assert held.needs_approval is True
    assert sandbox.calls == []
    held.approval = ApprovalDecision.APPROVAL_DECISION_APPROVED
    result = agent.tool_executor.execute_held_tool_call(held)

    assert result.is_completed()
    assert result.output == "sandboxed"
    [(tool, inputs, workspace)] = sandbox.calls
    assert tool.name == "skill__script_runner__shell"
    assert inputs == {"command": "bash scripts/check.sh", "cwd": "."}
    assert workspace.root == skill_dir.resolve()


def test_agent_rejects_duplicate_skill_names_before_registering_tools(tmp_path):
    first = _write_skill(tmp_path, "first", "duplicate")
    second = _write_skill(tmp_path, "second", "duplicate")
    catalog = DefaultToolRegistry()

    with pytest.raises(ValueError, match="duplicate skill name"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["unused"]),
            tool_catalog=catalog,
            skill_directories=[first, second],
        )

    assert catalog.get_all_tools() == []


def test_agent_rejects_skill_tool_collision_without_mutating_catalog(tmp_path):
    skill_dir = _write_skill(tmp_path, "research", "research helper")
    catalog = DefaultToolRegistry()
    existing = Tool(
        id="skill__research_helper__read_file",
        name="skill__research_helper__read_file",
        description="existing",
        input_schema={},
        executer=lambda _inputs: "existing",
    )
    catalog.register(existing)

    with pytest.raises(ValueError, match="tool collision"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["unused"]),
            tool_catalog=catalog,
            skill_directories=[skill_dir],
        )

    assert catalog.get_all_tools() == [existing]


def test_agent_rolls_back_skill_tools_when_catalog_registration_fails(tmp_path):
    class FailingCatalog(DefaultToolRegistry):
        def __init__(self):
            super().__init__()
            self.register_calls = 0

        def register(self, tool):
            self.register_calls += 1
            if self.register_calls == 2:
                raise RuntimeError("catalog offline")
            return super().register(tool)

    skill_dir = _write_skill(tmp_path, "scripts", "script runner")
    catalog = FailingCatalog()

    with pytest.raises(RuntimeError, match="catalog offline"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["unused"]),
            tool_catalog=catalog,
            sandbox_runner=RecordingSandbox(),
            skill_directories=[skill_dir],
        )

    assert catalog.get_all_tools() == []


def test_agent_rolls_back_skill_tools_from_a_catalog_that_stores_copies(tmp_path):
    class CopyingCatalog(DefaultToolRegistry):
        def __init__(self):
            super().__init__()
            self.register_calls = 0

        def register(self, tool):
            self.register_calls += 1
            if self.register_calls == 2:
                raise RuntimeError("catalog offline")
            return super().register(replace(tool))

    skill_dir = _write_skill(tmp_path, "scripts", "script runner")
    catalog = CopyingCatalog()

    with pytest.raises(RuntimeError, match="catalog offline"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["unused"]),
            tool_catalog=catalog,
            sandbox_runner=RecordingSandbox(),
            skill_directories=[skill_dir],
        )

    assert catalog.get_all_tools() == []


def test_agent_rolls_back_a_skill_tool_inserted_before_registration_raises(tmp_path):
    class InsertThenFailCatalog(DefaultToolRegistry):
        def __init__(self):
            super().__init__()
            self.register_calls = 0

        def register(self, tool):
            self.register_calls += 1
            super().register(tool)
            if self.register_calls == 2:
                raise RuntimeError("catalog offline")

    skill_dir = _write_skill(tmp_path, "scripts", "script runner")
    catalog = InsertThenFailCatalog()

    with pytest.raises(RuntimeError, match="catalog offline"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["unused"]),
            tool_catalog=catalog,
            sandbox_runner=RecordingSandbox(),
            skill_directories=[skill_dir],
        )

    assert catalog.get_all_tools() == []


def test_injected_executor_catalog_becomes_authoritative():
    catalog = DefaultToolRegistry()
    executor = DefaultToolExecutor(catalog)

    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
        tool_executor=executor,
    )

    assert agent.tool_registry is catalog
    assert agent.tool_executor is executor
    assert agent.mcp_manager.catalog is catalog
    assert catalog.get_all_tools() == []


def test_matching_catalog_executor_and_manager_are_reused():
    catalog = DefaultToolRegistry()
    executor = DefaultToolExecutor(catalog)
    manager = RecordingMCPManager(catalog)

    agent = Agent(
        config=Config(llm="test-model", max_steps=3, max_seconds=30),
        llm_gateway=ChatGateway(["hello"]),
        tool_catalog=catalog,
        tool_executor=executor,
        mcp_manager=manager,
    )

    assert agent.tool_registry is catalog
    assert agent.tool_executor is executor
    assert agent.mcp_manager is manager


def test_rejects_catalog_executor_mismatch_before_registering_local_tools():
    catalog = DefaultToolRegistry()
    executor_catalog = DefaultToolRegistry()

    with pytest.raises(ValueError, match="catalog"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["hello"]),
            tool_catalog=catalog,
            tool_executor=DefaultToolExecutor(executor_catalog),
        )

    assert catalog.get_all_tools() == []
    assert executor_catalog.get_all_tools() == []


def test_rejects_catalog_manager_mismatch_before_registering_local_tools():
    catalog = DefaultToolRegistry()
    manager_catalog = DefaultToolRegistry()

    with pytest.raises(ValueError, match="catalog"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["hello"]),
            tool_catalog=catalog,
            mcp_manager=RecordingMCPManager(manager_catalog),
        )

    assert catalog.get_all_tools() == []
    assert manager_catalog.get_all_tools() == []


def test_rejects_executor_manager_catalog_mismatch():
    executor_catalog = DefaultToolRegistry()
    manager_catalog = DefaultToolRegistry()

    with pytest.raises(ValueError, match="catalog"):
        Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["hello"]),
            tool_executor=DefaultToolExecutor(executor_catalog),
            mcp_manager=RecordingMCPManager(manager_catalog),
        )

    assert executor_catalog.get_all_tools() == []
    assert manager_catalog.get_all_tools() == []


def test_agent_mcp_methods_delegate_without_sync_loop_creation():
    async def scenario():
        catalog = DefaultToolRegistry()
        manager = RecordingMCPManager(catalog)
        agent = Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["hello"]),
            tool_catalog=catalog,
            mcp_manager=manager,
        )

        await agent.add_mcp_server("weather", "memory://weather", timeout=2)
        await agent.refresh_mcp_server("weather")
        await agent.remove_mcp_server("weather")

        assert manager.calls == [
            ("add", "weather", "memory://weather", {"timeout": 2}),
            ("refresh", "weather"),
            ("remove", "weather"),
        ]

    asyncio.run(scenario())


def test_agent_context_exit_closes_manager_once():
    async def scenario():
        catalog = DefaultToolRegistry()
        manager = RecordingMCPManager(catalog)
        async with Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=ChatGateway(["hello"]),
            tool_catalog=catalog,
            mcp_manager=manager,
        ) as agent:
            assert agent.mcp_manager is manager

        await agent.aclose()
        assert manager.close_calls == 1

    asyncio.run(scenario())


def test_closed_agent_rejects_chat_and_mcp_lifecycle_operations():
    async def scenario():
        catalog = DefaultToolRegistry()
        manager = RecordingMCPManager(catalog)
        gateway = ChatGateway(["unused"])
        agent = Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=gateway,
            tool_catalog=catalog,
            mcp_manager=manager,
        )
        await agent.aclose()

        with pytest.raises(RuntimeError, match="closed"):
            await agent.achat("hello")
        with pytest.raises(RuntimeError, match="closed"):
            await agent.add_mcp_server("server", "source")
        with pytest.raises(RuntimeError, match="closed"):
            await agent.refresh_mcp_server("server")
        with pytest.raises(RuntimeError, match="closed"):
            await agent.remove_mcp_server("server")

        assert gateway.plans == []

    asyncio.run(scenario())


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
