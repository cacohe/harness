from __future__ import annotations

from collections.abc import Mapping

from src.core.tools.contracts import ToolGateway, ToolRequest, ToolResult, ToolResultStatus


class ToolRouter:
    """Route reserved tool names without coupling independent gateways."""

    def __init__(
        self,
        routes: Mapping[str, ToolGateway],
        *,
        fallback: ToolGateway | None = None,
    ) -> None:
        if not isinstance(routes, Mapping):
            raise ValueError("Tool routes are invalid")
        snapshot: dict[str, ToolGateway] = {}
        for name, gateway in routes.items():
            if type(name) is not str or not name or not callable(getattr(gateway, "execute", None)):
                raise ValueError("Tool routes are invalid")
            snapshot[name] = gateway
        if fallback is not None and not callable(getattr(fallback, "execute", None)):
            raise ValueError("Tool fallback is invalid")
        self._routes = snapshot
        self._fallback = fallback

    async def execute(self, request: ToolRequest) -> ToolResult:
        if type(request) is not ToolRequest:
            raise ValueError("Tool request is invalid")
        gateway = self._routes.get(request.name)
        if gateway is None:
            gateway = self._fallback
        if gateway is None:
            return ToolResult(ToolResultStatus.DENIED, execution_key=request.idempotency_key)
        return await gateway.execute(request)
