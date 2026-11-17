import asyncio
import sqlite3
from pathlib import Path

import pytest
from src.adapters.tools.execution_store import (
    SQLiteToolExecutionStore,
    ToolBindingConflict,
    ToolExecutionStoreError,
)
from src.core.commands import ToolCallRequest
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import RunId, SessionId
from src.core.tools import ToolApproval, ToolRequest, ToolResult, ToolResultStatus


def _request(arguments: str = "{}") -> ToolRequest:
    call = ToolCallRequest(
        "call", "tool", arguments,
        requires_approval=False, approval_scope="execute", workspace_id="default",
    )
    return ToolRequest.from_call(
        call, run_id=RunId("run"), session_id=SessionId("session"),
        step_id="step", approval=ToolApproval.NOT_REQUIRED,
    )


def _result(request: ToolRequest) -> ToolResult:
    return ToolResult(
        ToolResultStatus.FAILED, failure=RunFailure(RunErrorCode.TOOL_FAILED),
        execution_key=request.idempotency_key,
    )


def test_completed_result_survives_reopen(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        request = _request()
        first = SQLiteToolExecutionStore(path)
        assert await first.claim(request) is None
        await first.complete(request, _result(request))
        reopened = SQLiteToolExecutionStore(path)
        assert await reopened.claim(request) == _result(request)

    asyncio.run(scenario())


def test_independent_instances_coordinate_same_active_key(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        first = SQLiteToolExecutionStore(path, timeout_seconds=1)
        second = SQLiteToolExecutionStore(path, timeout_seconds=1)
        request = _request()
        assert await first.claim(request) is None
        waiter = asyncio.create_task(second.claim(request))
        await asyncio.sleep(0.03)
        assert not waiter.done()
        await first.complete(request, _result(request))
        assert await waiter == _result(request)

    asyncio.run(scenario())


def test_started_is_recovered_without_replay(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        request = _request()
        assert await SQLiteToolExecutionStore(path).claim(request) is None
        recovered = await SQLiteToolExecutionStore(path).claim(request)
        assert recovered is not None and recovered.status is ToolResultStatus.FAILED
        assert recovered.execution_key == request.idempotency_key

    asyncio.run(scenario())


def test_binding_conflict_and_unknown_schema_are_rejected(tmp_path: Path) -> None:
    async def conflict():
        path = (tmp_path / "store.db").resolve()
        store = SQLiteToolExecutionStore(path)
        request = _request()
        assert await store.claim(request) is None
        await store.complete(request, _result(request))
        with pytest.raises(ToolBindingConflict):
            await store.claim(_request('{"changed":true}'))

    asyncio.run(conflict())
    unknown = (tmp_path / "unknown.db").resolve()
    with sqlite3.connect(unknown) as connection:
        connection.execute("CREATE TABLE alien(value TEXT) STRICT")
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(ToolExecutionStoreError):
        SQLiteToolExecutionStore(unknown)


def test_corrupt_completed_payload_is_rejected(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        request = _request()
        store = SQLiteToolExecutionStore(path)
        assert await store.claim(request) is None
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE executions SET state='completed', result_json='{}'"
            )
        with pytest.raises(ToolExecutionStoreError):
            await store.claim(request)

    asyncio.run(scenario())
