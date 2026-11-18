import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.tools import (
    InMemoryToolExecutionStore,
    SQLiteToolExecutionStore,
    ToolCatalog,
    ToolCatalogEntry,
)
from src.adapters.tools.local import LocalToolGateway
from src.core.commands import ToolCallRequest
from src.core.ids import RunId, SessionId
from src.core.tools import ToolApproval, ToolRequest, ToolSpec


def _request(arguments: str = '{"value":1}') -> ToolRequest:
    call = ToolCallRequest(
        "call-1",
        "increment",
        arguments,
        requires_approval=False,
        approval_scope="execute",
        workspace_id="default",
    )
    return ToolRequest.from_call(
        call,
        run_id=RunId("run-1"),
        session_id=SessionId("session-1"),
        step_id="step-1",
        approval=ToolApproval.NOT_REQUIRED,
    )


def test_concurrent_retry_has_one_effect_and_recovers_from_sqlite(tmp_path: Path) -> None:
    effects = 0

    async def handler(arguments):
        nonlocal effects
        effects += 1
        await asyncio.sleep(0)
        return {"value": arguments["value"] + 1}

    async def scenario():
        store = SQLiteToolExecutionStore((tmp_path / "tools.db").resolve())
        repo = InMemoryArtifactRepository(1_048_576)
        externalizer = PolicyArtifactExternalizer(
            repo,
            ArtifactPolicy(1024, 1_048_576, 1024, frozenset({"application/json"})),
        )
        spec = ToolSpec("increment", "increment", '{"type":"object"}')
        gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(spec, handler, False),)), store, externalizer
        )
        first, second = await asyncio.gather(
            gateway.execute(_request()), gateway.execute(_request())
        )
        assert first == second
        assert first.execution_key == _request().idempotency_key
        recovered = await gateway.execute(_request())
        assert recovered == first

    asyncio.run(scenario())
    assert effects == 1


def _externalizer(inline: int = 1024):
    return PolicyArtifactExternalizer(
        InMemoryArtifactRepository(1_048_576),
        ArtifactPolicy(inline, 1_048_576, 1024, frozenset({"application/json"})),
    )


def test_live_approval_tightening_prevents_claim_and_effect(tmp_path: Path) -> None:
    effects = 0

    async def handler(_arguments):
        nonlocal effects
        effects += 1
        return {}

    async def scenario():
        store = SQLiteToolExecutionStore((tmp_path / "tools.db").resolve())
        spec = ToolSpec("increment", "x", "true", requires_approval=True)
        gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(spec, handler, True),)), store, _externalizer()
        )
        result = await gateway.execute(_request())
        assert result.status.value == "pending_approval"
        assert result.execution_key == _request().idempotency_key

    asyncio.run(scenario())
    assert effects == 0


def test_sync_handler_runs_in_thread_and_large_result_externalizes(tmp_path: Path) -> None:
    caller = threading.get_ident()
    worker = caller

    def handler(_arguments):
        nonlocal worker
        worker = threading.get_ident()
        return {"content": "x" * 4096}

    async def scenario():
        gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), handler, False),)),
            SQLiteToolExecutionStore((tmp_path / "tools.db").resolve()),
            _externalizer(32),
        )
        result = await gateway.execute(_request())
        assert result.inline_content is None
        assert len(result.artifacts) == 1
        assert result.execution_key == _request().idempotency_key

    asyncio.run(scenario())
    assert worker != caller


def test_timeout_exception_and_cancellation_are_stable(tmp_path: Path) -> None:
    async def slow(_arguments):
        await asyncio.sleep(10)

    async def broken(_arguments):
        raise RuntimeError("host path F:/private and secret")

    async def scenario():
        slow_gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), slow, True),)),
            SQLiteToolExecutionStore(
                (tmp_path / "slow.db").resolve(), sqlite_busy_timeout_seconds=0.05
            ),
            _externalizer(),
            timeout_seconds=0.01,
        )
        timed = await slow_gateway.execute(_request())
        assert timed.status.value == "timed_out"
        assert timed.failure is not None and "private" not in timed.failure.message

        broken_gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), broken, True),)),
            SQLiteToolExecutionStore((tmp_path / "broken.db").resolve()),
            _externalizer(),
        )
        failed = await broken_gateway.execute(_request())
        assert failed.status.value == "failed"
        assert failed.failure is not None and "secret" not in failed.failure.message

        started = asyncio.Event()

        async def cancellable(_arguments):
            started.set()
            await asyncio.sleep(10)

        cancel_gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), cancellable, True),)),
            SQLiteToolExecutionStore(
                (tmp_path / "cancel.db").resolve(), sqlite_busy_timeout_seconds=0.05
            ),
            _externalizer(),
            timeout_seconds=20,
        )
        task = asyncio.create_task(cancel_gateway.execute(_request()))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("caller cancellation was swallowed")
        retry = await cancel_gateway.execute(_request())
        assert retry.status.value == "cancelled"

    asyncio.run(scenario())


def test_binding_change_is_rejected(tmp_path: Path) -> None:
    async def handler(arguments):
        return arguments

    async def scenario():
        store = SQLiteToolExecutionStore((tmp_path / "tools.db").resolve())
        externalizer = PolicyArtifactExternalizer(
            InMemoryArtifactRepository(1_048_576),
            ArtifactPolicy(1024, 1_048_576, 1024, frozenset({"application/json"})),
        )
        gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), handler, False),)),
            store,
            externalizer,
        )
        await gateway.execute(_request())
        try:
            await gateway.execute(_request('{"value":2}'))
        except RuntimeError:
            return
        raise AssertionError("binding conflict was accepted")

    asyncio.run(scenario())


def test_sync_mutating_handler_is_rejected_at_registration() -> None:
    def handler(_arguments):
        return {}

    with pytest.raises(ValueError, match="Mutating local tools"):
        LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), handler, True),)),
            InMemoryToolExecutionStore(),
            _externalizer(),
        )


def test_stale_workspace_binding_is_denied_before_claim(tmp_path: Path) -> None:
    async def handler(_arguments):
        raise AssertionError("stale binding executed")

    async def scenario():
        path = (tmp_path / "tools.db").resolve()
        store = SQLiteToolExecutionStore(path)
        gateway = LocalToolGateway(
            ToolCatalog((ToolCatalogEntry(ToolSpec("increment", "x", "true"), handler, False),)),
            store,
            _externalizer(),
        )
        call = ToolCallRequest(
            "call-1",
            "increment",
            "{}",
            requires_approval=False,
            approval_scope="execute",
            workspace_id="stale",
        )
        request = ToolRequest.from_call(
            call,
            run_id=RunId("run-1"),
            session_id=SessionId("session-1"),
            step_id="step-1",
            approval=ToolApproval.NOT_REQUIRED,
        )
        result = await gateway.execute(request)
        assert result.status.value == "denied"
        assert result.execution_key == request.idempotency_key
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT count(*) FROM executions").fetchone() == (0,)

    asyncio.run(scenario())
