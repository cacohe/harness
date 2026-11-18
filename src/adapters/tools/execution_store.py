from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.core.artifacts import ArtifactRef, InlineContent
from src.core.errors import RunErrorCode, RunFailure
from src.core.tools import ToolRequest, ToolResult, ToolResultStatus

_SCHEMA_VERSION = 1
_DDL = """
CREATE TABLE executions (
  session_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  tool_call_id TEXT NOT NULL,
  binding_key TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('started', 'completed')),
  result_json TEXT,
  PRIMARY KEY(session_id, run_id, tool_call_id)
) STRICT;
"""


class ToolExecutionStoreError(RuntimeError):
    pass


class ToolBindingConflict(ToolExecutionStoreError):
    pass


class ToolExecutionStore(Protocol):
    async def claim(self, request: ToolRequest) -> ToolResult | None: ...

    async def complete(self, request: ToolRequest, result: ToolResult) -> None: ...


@dataclass(slots=True)
class InMemoryToolExecutionStore:
    """Explicitly development-only execution store."""

    _records: dict[tuple[str, str, str], tuple[str, ToolResult | None]] = field(
        init=False, default_factory=dict
    )
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    async def claim(self, request: ToolRequest) -> ToolResult | None:
        key = _identity(request)
        async with self._lock:
            current = self._records.get(key)
            if current is None:
                self._records[key] = (request.idempotency_key, None)
                return None
            binding, result = current
            if binding != request.idempotency_key:
                raise ToolBindingConflict("Tool call identity is already bound")
            return result if result is not None else _interrupted(request.idempotency_key)

    async def complete(self, request: ToolRequest, result: ToolResult) -> None:
        key = _identity(request)
        async with self._lock:
            current = self._records.get(key)
            if current is None or current[0] != request.idempotency_key:
                raise ToolBindingConflict("Tool execution claim is missing or changed")
            if current[1] is not None and current[1] != result:
                raise ToolExecutionStoreError("Tool execution already completed differently")
            self._records[key] = (request.idempotency_key, result)


class SQLiteToolExecutionStore:
    """Strict SQLite store; startup recovery never replays an unknown side effect."""

    def __init__(self, path: Path, timeout_seconds: float = 5.0) -> None:
        if type(path) is not type(Path()) or not path.is_absolute():
            raise ValueError("Tool execution store requires an absolute SQLite path")
        if type(timeout_seconds) not in (int, float) or isinstance(timeout_seconds, bool):
            raise ValueError("Tool execution store timeout is invalid")
        self._path = Path(path)
        self._timeout = float(timeout_seconds)
        if not 0 < self._timeout <= 300:
            raise ValueError("Tool execution store timeout is invalid")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=self._timeout)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if version == 0 and not tables:
                connection.executescript(_DDL)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION or tables != {"executions"}:
                raise ToolExecutionStoreError("Unknown tool execution store schema")
            columns = tuple(
                row[1] for row in connection.execute("PRAGMA table_info(executions)")
            )
            if columns != (
                "session_id", "run_id", "tool_call_id", "binding_key", "state", "result_json"
            ):
                raise ToolExecutionStoreError("Corrupt tool execution store schema")
            rows = connection.execute(
                "SELECT session_id, run_id, tool_call_id, binding_key FROM executions "
                "WHERE state='started'"
            ).fetchall()
            for session_id, run_id, call_id, binding in rows:
                connection.execute(
                    "UPDATE executions SET state='completed', result_json=? "
                    "WHERE session_id=? AND run_id=? AND tool_call_id=? AND state='started'",
                    (_encode_result(_interrupted(binding)), session_id, run_id, call_id),
                )

    async def claim(self, request: ToolRequest) -> ToolResult | None:
        return await asyncio.to_thread(self._claim, request)

    def _claim(self, request: ToolRequest) -> ToolResult | None:
        identity = _identity(request)
        deadline = time.monotonic() + self._timeout
        while True:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT binding_key, state, result_json FROM executions "
                    "WHERE session_id=? AND run_id=? AND tool_call_id=?", identity
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO executions VALUES (?, ?, ?, ?, 'started', NULL)",
                        (*identity, request.idempotency_key),
                    )
                    return None
                binding, state, encoded = row
                if binding != request.idempotency_key:
                    raise ToolBindingConflict("Tool call identity is already bound")
                if state == "completed":
                    if type(encoded) is not str:
                        raise ToolExecutionStoreError("Corrupt tool execution record")
                    return _decode_result(encoded, binding)
                if state != "started" or encoded is not None:
                    raise ToolExecutionStoreError("Corrupt tool execution record")
            if time.monotonic() >= deadline:
                interrupted = _interrupted(request.idempotency_key)
                self._complete(request, interrupted)
                return interrupted
            time.sleep(0.01)

    async def complete(self, request: ToolRequest, result: ToolResult) -> None:
        await asyncio.to_thread(self._complete, request, result)

    def _complete(self, request: ToolRequest, result: ToolResult) -> None:
        identity = _identity(request)
        encoded = _encode_result(result)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT binding_key, state, result_json FROM executions "
                "WHERE session_id=? AND run_id=? AND tool_call_id=?", identity
            ).fetchone()
            if row is None or row[0] != request.idempotency_key:
                raise ToolBindingConflict("Tool execution claim is missing or changed")
            if row[1] == "completed":
                if row[2] != encoded:
                    raise ToolExecutionStoreError("Tool execution already completed differently")
                return
            connection.execute(
                "UPDATE executions SET state='completed', result_json=? "
                "WHERE session_id=? AND run_id=? AND tool_call_id=?", (encoded, *identity)
            )


def _identity(request: ToolRequest) -> tuple[str, str, str]:
    return (request.session_id.value, request.run_id.value, request.tool_call_id)


def _interrupted(execution_key: str) -> ToolResult:
    return ToolResult(
        ToolResultStatus.FAILED,
        failure=RunFailure(RunErrorCode.TOOL_FAILED),
        execution_key=execution_key,
    )


def _encode_result(result: ToolResult) -> str:
    inline = result.inline_content
    payload: dict[str, object] = {
        "status": result.status.value,
        "execution_key": result.execution_key,
        "failure": result.failure.code.value if result.failure is not None else None,
        "inline": None,
        "artifacts": [],
        "summary": result.summary,
    }
    if inline is not None:
        payload["inline"] = {"media_type": inline.media_type, "value": inline.value}
    payload["artifacts"] = [
        {
            "artifact_id": ref.artifact_id, "media_type": ref.media_type,
            "size_bytes": ref.size_bytes, "sha256": ref.sha256,
            "name": ref.name, "metadata": ref.metadata,
        }
        for ref in result.artifacts
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decode_result(encoded: str, binding: str) -> ToolResult:
    try:
        value = json.loads(encoded)
        if type(value) is not dict or value.get("execution_key") != binding:
            raise ValueError
        status = ToolResultStatus(value["status"])
        failure_value = value.get("failure")
        failure = RunFailure(RunErrorCode(failure_value)) if failure_value is not None else None
        inline_value = value.get("inline")
        inline = None
        if inline_value is not None:
            if type(inline_value) is not dict:
                raise ValueError
            inline = InlineContent(inline_value["value"], inline_value["media_type"])
        raw_artifacts = value.get("artifacts", [])
        if type(raw_artifacts) is not list:
            raise ValueError
        artifacts = tuple(
            ArtifactRef(
                item["artifact_id"], item["media_type"], item["size_bytes"], item["sha256"],
                item.get("name"), item.get("metadata"),
            )
            for item in raw_artifacts
        )
        return ToolResult(
            status, inline_content=inline, artifacts=artifacts, summary=value.get("summary"),
            failure=failure, execution_key=binding,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ToolExecutionStoreError("Corrupt tool execution result") from None


# Public implementation moved to the stricter owner/wait protocol.
from src.adapters.tools.idempotency import (  # type: ignore[assignment] # noqa: E402,F811
    ClaimState,
    ExecutionClaim,
    InMemoryToolExecutionStore,
    SQLiteToolExecutionStore,
    ToolBindingConflict,
    ToolExecutionStore,
    ToolExecutionStoreError,
    ToolExecutionWaitExpired,
)

__all__ = [
    "ClaimState",
    "ExecutionClaim",
    "InMemoryToolExecutionStore",
    "SQLiteToolExecutionStore",
    "ToolBindingConflict",
    "ToolExecutionStore",
    "ToolExecutionStoreError",
    "ToolExecutionWaitExpired",
]
