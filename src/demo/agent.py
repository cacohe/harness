from src.config import Config
from src.context.context_builder import DefaultContextBuilder
from src.core.context import ContextBuilder
from src.core.guard import HarnessGuard
from src.core.llm import LlmGateway
from src.core.loop import DEFAULT_SYSTEM_PROMPT, Harness
from src.core.task import Task
from src.core.tool import ToolExecutor
from src.llm.litellm import LiteLLMGateway
from src.schema.hook import HarnessHook
from src.tool.local_tools import LOCAL_TOOLS
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry


class Agent:
    """One harness wired to LiteLLM, the default context builder, and the local tools."""

    def __init__(
        self,
        config: Config | None = None,
        llm_gateway: LlmGateway | None = None,
        tool_executor: ToolExecutor | None = None,
        context_builder: ContextBuilder | None = None,
    ):
        """
        Create an agent with a model gateway, a context builder, and the local tools.
        Args:
            config: Runtime config. When omitted, Config.load reads the environment and files.
            llm_gateway: Gateway to use. When omitted, a LiteLLM gateway is created from config.
            tool_executor: Executor to use. When omitted, the local tools are registered.
            context_builder: Builder to use. When omitted, the default builder is created.
        """
        self.config = config or Config.load()
        self.transcript: list[tuple[str, str]] = []
        self.tool_registry = DefaultToolRegistry()
        self.tool_registry.tools = {}
        for tool in LOCAL_TOOLS:
            self.tool_registry.register(tool)
        self.llm_gateway = llm_gateway or LiteLLMGateway(
            model=self.config.llm,
            api_key=self.config.llm_api_key,
            api_base=self.config.llm_api_base,
        )
        self.tool_executor = tool_executor or DefaultToolExecutor(self.tool_registry)
        self.context_builder = context_builder or DefaultContextBuilder()
        self.harness = Harness(
            DEFAULT_SYSTEM_PROMPT,
            self.llm_gateway,
            self.tool_registry.get_all_tools(),
            [],
            self.tool_executor,
            self.context_builder,
            harness_hooks=HarnessHook(),
            harness_guard=HarnessGuard(),
            max_steps=self.config.max_steps,
            max_seconds=self.config.max_seconds,
        )

    def run(self, input: str) -> Task:
        """
        Run one task with the registered local tools.
        Args:
            input: The user's request.
        Returns:
            The task after the harness stops. Its state records completion or failure.
        """
        task = Task(original_input=input)
        self.harness.run(task)
        return task

    def chat(self, message: str) -> Task:
        """
        Plan one message, run it to an answer, and remember the turn.
        Args:
            message: The user's next line.
        Returns:
            The task after planning and execution. A failed turn is not remembered.
        """
        task = Task(original_input=self._with_transcript(message))
        self.harness.run(task)
        if task.is_running():
            self.harness.run(task)
        if task.is_completed() and task.result is not None:
            self.transcript.append(("user", message))
            self.transcript.append(("assistant", task.result.output))
        return task

    def _with_transcript(self, message: str) -> str:
        if not self.transcript:
            return message
        prior = "\n".join(f"{role}: {content}" for role, content in self.transcript)
        return f"{prior}\nuser: {message}"

    def to_dict(self) -> dict:
        """
        Describe the wired agent.
        Returns:
            The model name and the registered tool names.
        """
        return {
            "model": getattr(self.llm_gateway, "model", None),
            "tools": [tool.name for tool in self.tool_registry.get_all_tools()],
        }
