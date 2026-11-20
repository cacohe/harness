import asyncio

from src.adapters.tools.router import ToolRouter
from src.core.commands import ToolCallRequest
from src.core.ids import RunId, SessionId
from src.core.tools import ToolApproval, ToolRequest, ToolResult, ToolResultStatus


class _Gateway:
    def __init__(self, status: ToolResultStatus) -> None:
        self.status = status
        self.names: list[str] = []

    async def execute(self, request: ToolRequest) -> ToolResult:
        self.names.append(request.name)
        return ToolResult(self.status, execution_key=request.idempotency_key)


def _request(name: str) -> ToolRequest:
    call = ToolCallRequest(
        "call-1",
        name,
        "{}",
        requires_approval=False,
        approval_scope="execute",
        workspace_id="workspace",
    )
    return ToolRequest.from_call(
        call,
        run_id=RunId("run-1"),
        session_id=SessionId("session-1"),
        step_id="step-1",
        approval=ToolApproval.NOT_REQUIRED,
    )


def test_router_selects_reserved_gateway_and_falls_back_for_other_names() -> None:
    async def scenario() -> None:
        reserved = _Gateway(ToolResultStatus.DENIED)
        fallback = _Gateway(ToolResultStatus.PENDING_APPROVAL)
        router = ToolRouter({"read_artifact": reserved}, fallback=fallback)

        read = await router.execute(_request("read_artifact"))
        custom = await router.execute(_request("custom"))

        assert read.status is ToolResultStatus.DENIED
        assert custom.status is ToolResultStatus.PENDING_APPROVAL
        assert reserved.names == ["read_artifact"]
        assert fallback.names == ["custom"]

    asyncio.run(scenario())


def test_router_denies_unknown_tool_without_fallback() -> None:
    async def scenario() -> None:
        router = ToolRouter({})
        request = _request("unknown")
        result = await router.execute(request)
        assert result.status is ToolResultStatus.DENIED
        assert result.execution_key == request.idempotency_key

    asyncio.run(scenario())
