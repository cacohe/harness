from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable

from src.adapters.tools.catalog import ToolCatalog, ToolCatalogEntry
from src.adapters.tools.execution_store import ClaimState, ToolExecutionStore
from src.core.artifacts import ArtifactExternalizer
from src.core.errors import RunErrorCode, RunFailure
from src.core.tools import ToolApproval, ToolRequest, ToolResult, ToolResultStatus


class LocalToolGateway:
    """Validated local dispatch with durable claim-before-effect semantics."""

    def __init__(
        self,
        catalog: ToolCatalog,
        store: ToolExecutionStore,
        externalizer: ArtifactExternalizer,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        if type(catalog) is not ToolCatalog:
            raise ValueError("Local tool catalog is invalid")
        if isinstance(timeout_seconds, bool) or type(timeout_seconds) not in (int, float):
            raise ValueError("Local tool timeout is invalid")
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("Local tool timeout is invalid")
        self._catalog = catalog
        if type(self) is LocalToolGateway and any(
            entry.mutating and not inspect.iscoroutinefunction(entry.handler)
            for entry in catalog.entries
        ):
            raise ValueError("Mutating local tools require cancellation-aware async handlers")
        self._store = store
        self._externalizer = externalizer
        self._timeout = timeout
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def execute(self, request: ToolRequest) -> ToolResult:
        entry = self._catalog.get(request.name)
        decision = _authorize(request, entry)
        if decision is not None:
            return decision
        assert entry is not None
        arguments = _arguments(request.arguments_json, entry)
        identity = (request.session_id.value, request.run_id.value, request.tool_call_id)
        async with self._locks_guard:
            lock = self._locks.setdefault(identity, asyncio.Lock())
        try:
            async with lock:
                claim = await self._store.claim(request)
                if claim.state is ClaimState.COMPLETED:
                    assert claim.result is not None
                    return claim.result
                if claim.state is ClaimState.WAIT:
                    return await self._store.wait_completed(request)
                try:
                    result = await self._invoke(request, entry.handler, arguments)
                except asyncio.CancelledError:
                    cancelled = ToolResult(
                        ToolResultStatus.CANCELLED,
                        failure=RunFailure(RunErrorCode.CANCELLED),
                        execution_key=request.idempotency_key,
                    )
                    completion = asyncio.create_task(self._store.complete(request, cancelled))
                    while not completion.done():
                        try:
                            await asyncio.shield(completion)
                        except asyncio.CancelledError:
                            continue
                    await completion
                    raise
                await self._store.complete(request, result)
                return result
        finally:
            async with self._locks_guard:
                if not lock.locked():
                    self._locks.pop(identity, None)

    async def _invoke(
        self,
        request: ToolRequest,
        handler: Callable[[dict[str, object]], object | Awaitable[object]],
        arguments: dict[str, object],
    ) -> ToolResult:
        try:
            if inspect.iscoroutinefunction(handler):
                value = await asyncio.wait_for(handler(arguments), self._timeout)
            else:
                value = await asyncio.wait_for(asyncio.to_thread(handler, arguments), self._timeout)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, self._timeout)
            content = await self._externalizer.externalize(value, "application/json")
            return ToolResult(
                ToolResultStatus.SUCCESS,
                inline_content=content.inline_content,
                artifacts=content.artifacts,
                summary=content.summary,
                execution_key=request.idempotency_key,
            )
        except TimeoutError:
            return ToolResult(
                ToolResultStatus.TIMED_OUT,
                failure=RunFailure(RunErrorCode.TOOL_TIMED_OUT),
                execution_key=request.idempotency_key,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolResult(
                ToolResultStatus.FAILED,
                failure=RunFailure(RunErrorCode.TOOL_FAILED),
                execution_key=request.idempotency_key,
            )


def _authorize(request: ToolRequest, entry: ToolCatalogEntry | None) -> ToolResult | None:
    if request.approval is ToolApproval.DENIED:
        return ToolResult(ToolResultStatus.DENIED, execution_key=request.idempotency_key)
    if request.approval is ToolApproval.UNDECIDED:
        return ToolResult(ToolResultStatus.PENDING_APPROVAL, execution_key=request.idempotency_key)
    if entry is None:
        return ToolResult(ToolResultStatus.DENIED, execution_key=request.idempotency_key)
    spec = entry.spec
    if spec.name != request.name or spec.approval_scope != request.approval_scope:
        return ToolResult(ToolResultStatus.DENIED, execution_key=request.idempotency_key)
    if spec.workspace_id != request.workspace_id:
        return ToolResult(ToolResultStatus.DENIED, execution_key=request.idempotency_key)
    # A live policy may tighten, but it can never remove the persisted gate.
    if spec.requires_approval and request.approval is not ToolApproval.APPROVED:
        return ToolResult(ToolResultStatus.PENDING_APPROVAL, execution_key=request.idempotency_key)
    if request.approval_required and request.approval is not ToolApproval.APPROVED:
        return ToolResult(ToolResultStatus.PENDING_APPROVAL, execution_key=request.idempotency_key)
    return None


def _arguments(encoded: str, entry: ToolCatalogEntry) -> dict[str, object]:
    try:
        arguments = json.loads(encoded)
        schema = json.loads(entry.spec.input_schema_json)
    except (json.JSONDecodeError, RecursionError):
        raise ValueError("Tool arguments are invalid") from None
    if type(arguments) is not dict or type(schema) not in (dict, bool):
        raise ValueError("Tool arguments must be a JSON object")
    if schema is False:
        raise ValueError("Tool schema rejects all arguments")
    if schema is True:
        return arguments
    if schema.get("type") not in (None, "object"):
        raise ValueError("Tool schema must accept an object")
    required = schema.get("required", [])
    if type(required) is not list or any(type(item) is not str for item in required):
        raise ValueError("Tool schema is invalid")
    if any(item not in arguments for item in required):
        raise ValueError("Tool arguments do not satisfy the schema")
    properties = schema.get("properties", {})
    if type(properties) is not dict:
        raise ValueError("Tool schema is invalid")
    if schema.get("additionalProperties") is False and any(
        key not in properties for key in arguments
    ):
        raise ValueError("Tool arguments do not satisfy the schema")
    for key, value in arguments.items():
        descriptor = properties.get(key)
        if type(descriptor) is dict and not _matches_type(value, descriptor.get("type")):
            raise ValueError("Tool arguments do not satisfy the schema")
    return arguments


def _matches_type(value: object, expected: object) -> bool:
    if expected is None:
        return True
    types = expected if type(expected) is list else [expected]
    mapping: dict[str, type[object] | tuple[type[object], ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
        "null": type(None),
    }
    for item in types:
        if type(item) is not str:
            continue
        target = mapping.get(item)
        if target is not None and isinstance(value, target):
            if item in {"integer", "number"} and isinstance(value, bool):
                continue
            return True
    return False
