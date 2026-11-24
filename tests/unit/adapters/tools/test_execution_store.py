import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest
from src.adapters.tools.execution_store import (
    ClaimState,
    SQLiteToolExecutionStore,
    ToolBindingConflict,
    ToolExecutionStoreError,
    ToolExecutionWaitExpired,
)
from src.core.execution.commands import ToolCallRequest
from src.core.execution.errors import RunErrorCode, RunFailure
from src.core.ids import RunId, SessionId
from src.core.tools.contracts import ToolApproval, ToolRequest, ToolResult, ToolResultStatus


def _request(arguments: str = "{}", *, call_id: str = "call") -> ToolRequest:
    call = ToolCallRequest(
        call_id,
        "tool",
        arguments,
        requires_approval=False,
        approval_scope="execute",
        workspace_id="default",
    )
    return ToolRequest.from_call(
        call,
        run_id=RunId("run"),
        session_id=SessionId("session"),
        step_id="step",
        approval=ToolApproval.NOT_REQUIRED,
    )


def _result(request: ToolRequest) -> ToolResult:
    return ToolResult(
        ToolResultStatus.FAILED,
        failure=RunFailure(RunErrorCode.TOOL_FAILED),
        execution_key=request.idempotency_key,
    )


def test_completed_result_survives_reopen(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        request = _request()
        first = SQLiteToolExecutionStore(path)
        assert (await first.claim(request)).state is ClaimState.OWNER
        await first.complete(request, _result(request))
        reopened = SQLiteToolExecutionStore(path)
        assert (await reopened.claim(request)).result == _result(request)

    asyncio.run(scenario())


def test_wait_claim_is_read_only_under_reserved_writer_lock(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        store = SQLiteToolExecutionStore(
            path, sqlite_busy_timeout_seconds=0.001, poll_interval_seconds=0.001
        )
        request = _request()
        assert (await store.claim(request)).state is ClaimState.OWNER
        lock = sqlite3.connect(path, timeout=0.001)
        try:
            lock.execute("BEGIN IMMEDIATE")
            claims = await asyncio.gather(*(store.claim(request) for _ in range(12)))
            assert all(claim.state is ClaimState.WAIT for claim in claims)
        finally:
            lock.rollback()
            lock.close()

    asyncio.run(scenario())


def test_first_claim_write_race_is_stable_across_many_rounds(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        stores = [
            SQLiteToolExecutionStore(
                path, sqlite_busy_timeout_seconds=0.01, poll_interval_seconds=0.001
            )
            for _ in range(8)
        ]
        for index in range(20):
            request = _request(call_id=f"call-{index}")
            outcomes = await asyncio.gather(
                *(store.claim(request) for store in stores), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    assert type(outcome) is ToolExecutionStoreError
                    assert str(outcome) == "Tool execution store is busy"
            claims = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
            for store, outcome in zip(stores, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    claims.append(await store.claim(request))
            assert sum(claim.state is ClaimState.OWNER for claim in claims) == 1
            assert sum(claim.state is ClaimState.WAIT for claim in claims) == 7

    asyncio.run(scenario())


def test_independent_instances_coordinate_same_active_key(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        first = SQLiteToolExecutionStore(path, sqlite_busy_timeout_seconds=1)
        second = SQLiteToolExecutionStore(path, sqlite_busy_timeout_seconds=1)
        request = _request()
        assert (await first.claim(request)).state is ClaimState.OWNER
        waiter = asyncio.create_task(second.wait_completed(request))
        await asyncio.sleep(0.03)
        assert not waiter.done()
        await first.complete(request, _result(request))
        assert await waiter == _result(request)

    asyncio.run(scenario())


def test_three_instances_wait_longer_than_busy_timeout_without_poisoning(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        stores = [
            SQLiteToolExecutionStore(
                path, sqlite_busy_timeout_seconds=0.01, poll_interval_seconds=0.005
            )
            for _ in range(3)
        ]
        request = _request()
        assert (await stores[0].claim(request)).state is ClaimState.OWNER
        waiters = [asyncio.create_task(store.wait_completed(request)) for store in stores[1:]]
        await asyncio.sleep(0.08)
        assert all(not waiter.done() for waiter in waiters)
        await stores[0].complete(request, _result(request))
        assert await asyncio.gather(*waiters) == [_result(request), _result(request)]

    asyncio.run(scenario())


def test_wait_expiry_never_changes_live_owner_row(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        owner = SQLiteToolExecutionStore(path)
        contender = SQLiteToolExecutionStore(
            path, poll_interval_seconds=0.005, execution_wait_seconds=0.02
        )
        request = _request()
        assert (await owner.claim(request)).state is ClaimState.OWNER
        with pytest.raises(ToolExecutionWaitExpired):
            await contender.wait_completed(request)
        assert (await owner.claim(request)).state is ClaimState.WAIT
        await owner.complete(request, _result(request))
        assert (await contender.claim(request)).result == _result(request)

    asyncio.run(scenario())


def test_wrong_execution_key_is_zero_write(tmp_path: Path) -> None:
    async def scenario():
        store = SQLiteToolExecutionStore((tmp_path / "store.db").resolve())
        request = _request()
        assert (await store.claim(request)).state is ClaimState.OWNER
        wrong = ToolResult(
            ToolResultStatus.FAILED,
            failure=RunFailure(RunErrorCode.TOOL_FAILED),
            execution_key="0" * 64,
        )
        with pytest.raises(ToolExecutionStoreError):
            await store.complete(request, wrong)
        assert (await store.claim(request)).state is ClaimState.WAIT

    asyncio.run(scenario())


def test_started_is_recovered_without_replay(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "store.db").resolve()
        request = _request()
        store = SQLiteToolExecutionStore(path)
        assert (await store.claim(request)).state is ClaimState.OWNER
        reopened = SQLiteToolExecutionStore(path)
        assert (await reopened.claim(request)).state is ClaimState.WAIT
        assert await reopened.recover_abandoned() == 1
        recovered = (await reopened.claim(request)).result
        assert recovered is not None and recovered.status is ToolResultStatus.FAILED
        assert recovered.execution_key == request.idempotency_key

    asyncio.run(scenario())


def test_binding_conflict_and_unknown_schema_are_rejected(tmp_path: Path) -> None:
    async def conflict():
        path = (tmp_path / "store.db").resolve()
        store = SQLiteToolExecutionStore(path)
        request = _request()
        assert (await store.claim(request)).state is ClaimState.OWNER
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
        assert (await store.claim(request)).state is ClaimState.OWNER
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE executions SET state='completed', result_json='{}'")
        with pytest.raises(ToolExecutionStoreError):
            await store.claim(request)

    asyncio.run(scenario())


def test_every_operation_deterministically_closes_sqlite_connections(
    tmp_path: Path, monkeypatch
) -> None:
    class TrackingConnection:
        def __init__(self, connection):
            self.connection = connection
            self.closed = False

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def close(self):
            self.closed = True
            self.connection.close()

    async def scenario():
        store = SQLiteToolExecutionStore((tmp_path / "store.db").resolve())
        original_connect = store._connect
        opened = []

        def connect():
            tracked = TrackingConnection(original_connect())
            opened.append(tracked)
            return tracked

        monkeypatch.setattr(store, "_connect", connect)
        request = _request()
        assert (await store.claim(request)).state is ClaimState.OWNER
        assert opened and all(connection.closed for connection in opened)
        await store.complete(request, _result(request))
        assert all(connection.closed for connection in opened)
        assert (await store.claim(request)).state is ClaimState.COMPLETED
        assert all(connection.closed for connection in opened)
        with pytest.raises(ToolBindingConflict):
            await store.claim(_request('{"changed":true}'))
        assert all(connection.closed for connection in opened)

    asyncio.run(scenario())


def test_deep_result_json_maps_recursion_to_stable_corrupt_result_error(tmp_path: Path) -> None:
    async def scenario():
        path = (tmp_path / "deep.db").resolve()
        store = SQLiteToolExecutionStore(path)
        request = _request()
        assert (await store.claim(request)).state is ClaimState.OWNER
        depth = sys.getrecursionlimit() + 100
        encoded = "[" * depth + "0" + "]" * depth
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE executions SET state='completed', result_json=?",
                (encoded,),
            )
        with pytest.raises(
            ToolExecutionStoreError, match="^Corrupt tool execution result$"
        ) as caught:
            await store.claim(request)
        assert caught.value.__cause__ is None

    asyncio.run(scenario())
