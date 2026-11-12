from src.core.tool import (
    ApprovalPolicy,
    SandboxRunner,
    ToolCatalog,
    ToolExecutor,
    ToolRunner,
)
from src.schema.tool import ApprovalDecision, Tool


class LocalToolRunner:
    """Run the callable attached to a local tool."""

    def run(self, tool: Tool, inputs: dict):
        return tool.executer(inputs)


class DefaultApprovalPolicy:
    """Pause local tools that can execute commands or mutate files."""

    _NEED_APPROVAL_TOOLS = {"shell", "write_file", "edit_file"}

    def decide(self, tool: Tool, inputs: dict) -> ApprovalDecision | None:
        if tool.name in self._NEED_APPROVAL_TOOLS:
            return None
        return ApprovalDecision.APPROVAL_DECISION_APPROVED


class DefaultToolExecutor(ToolExecutor):
    """Tool executor wired to the local callable runner and approval policy."""

    def __init__(
        self,
        tool_registry: ToolCatalog,
        *,
        runner: ToolRunner | None = None,
        approval_policy: ApprovalPolicy | None = None,
        sandbox_runner: SandboxRunner | None = None,
    ) -> None:
        super().__init__(
            tool_registry,
            runner if runner is not None else LocalToolRunner(),
            approval_policy=(
                approval_policy if approval_policy is not None else DefaultApprovalPolicy()
            ),
            sandbox_runner=sandbox_runner,
        )
