from src.config import Config
from src.context.context_builder import DefaultContextBuilder
from src.core.context import ContextBuilder
from src.core.guard import NoOpGuard
from src.core.llm import LlmGateway
from src.core.loop import Harness
from src.core.task import Task
from src.core.tool import ToolExecutor
from src.llm.litellm import LiteLLMGateway
from src.schema.hook import HarnessHook
from src.schema.tool import ToolExecutionResult
from src.tool.local_tools import LOCAL_TOOLS
from src.tool.tool_executor import DefaultToolExecutor
from src.tool.tool_registry import DefaultToolRegistry

SYSTEM_PROMPT = "You are a helpful AI Agent that can help with tasks."


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
        for tool in LOCAL_TOOLS:
            self.tool_registry.register(tool)
        self.llm_gateway = llm_gateway or LiteLLMGateway(
            model=self.config.llm,
            api_key=self.config.llm_api_key,
            api_base=self.config.llm_api_base,
        )
        self.tool_executor = tool_executor or DefaultToolExecutor(self.tool_registry)
        self.context_builder = context_builder or DefaultContextBuilder()
        self.pending_task: Task | None = None
        self.pending_message: str | None = None
        self.harness = Harness(
            SYSTEM_PROMPT,
            self.llm_gateway,
            [],
            self.tool_executor,
            self.context_builder,
            harness_hooks=HarnessHook(),
            harness_guard=NoOpGuard(),
            max_steps=self.config.max_steps,
            max_seconds=self.config.max_seconds,
        )

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
        if task.is_paused():
            self.pending_task = task
            self.pending_message = message
            return task
        self.pending_task = None
        self.pending_message = None
        self._remember(message, task)
        return task

    def approval_requests(self) -> list[ToolExecutionResult]:
        """
        List the tool calls on the paused task that still wait for a decision.
        Returns:
            Undecided calls in call order. Empty when no task is paused.
        """
        if self.pending_task is None:
            return []
        return self.pending_task.approval_requests()

    def decide_tool_call(self, tool_id: str, approved: bool) -> None:
        """
        Approve or reject one tool call waiting on the paused task.
        Args:
            tool_id: The id of the pending tool call.
            approved: True runs the call on resume. False records it as rejected.
        Raises:
            ValueError: No task is waiting for approval.
        """
        if self.pending_task is None:
            raise ValueError("nothing is waiting for approval")
        self.pending_task.decide_tool_call(tool_id, approved)

    def resume(self) -> Task:
        """
        Continue the paused task after every pending call is decided.
        Returns:
            The task after it continues.
        Raises:
            ValueError: No task is waiting for approval.
        """
        task = self.pending_task
        if task is None:
            raise ValueError("nothing is waiting for approval")
        message = self.pending_message or ""
        try:
            self.harness.run(task)
        finally:
            if not task.is_paused():
                self.pending_task = None
                self.pending_message = None
        if not task.is_paused():
            self._remember(message, task)
        return task

    def _remember(self, message: str, task: Task) -> None:
        if task.is_completed() and task.result is not None:
            self.transcript.append(("user", message))
            self.transcript.append(("assistant", task.result.output))

    def _with_transcript(self, message: str) -> str:
        if not self.transcript:
            return message
        prior = "\n".join(f"{role}: {content}" for role, content in self.transcript)
        return f"{prior}\nuser: {message}"
