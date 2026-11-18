from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NoReturn, Protocol, cast

from src.core.artifacts import ArtifactRef, InlineContent
from src.core.errors import RunErrorCode, RunFailure
from src.core.tools import ToolRequest, ToolResult, ToolResultStatus, snapshot_tool_result

_VERSION = 1
_MAX_RESULT_BYTES = 2 * 1_048_576
_MAX_DEPTH = 24
_FINGERPRINT = "tool-execution-v1-strict-canonical"
_DDL_META = "CREATE TABLE schema_meta (singleton INTEGER PRIMARY KEY CHECK(singleton = 1), version INTEGER NOT NULL, fingerprint TEXT NOT NULL) STRICT"
_DDL_EXECUTIONS = "CREATE TABLE executions (session_id TEXT NOT NULL, run_id TEXT NOT NULL, tool_call_id TEXT NOT NULL, binding_key TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('started', 'completed')), result_json TEXT, CHECK((state = 'started' AND result_json IS NULL) OR (state = 'completed' AND result_json IS NOT NULL)), PRIMARY KEY(session_id, run_id, tool_call_id)) STRICT"
_EXPECTED_COLUMNS = {
    "schema_meta": (("singleton", "INTEGER", 0, 1), ("version", "INTEGER", 1, 0), ("fingerprint", "TEXT", 1, 0)),
    "executions": (
        ("session_id", "TEXT", 1, 1), ("run_id", "TEXT", 1, 2),
        ("tool_call_id", "TEXT", 1, 3), ("binding_key", "TEXT", 1, 0),
        ("state", "TEXT", 1, 0), ("result_json", "TEXT", 0, 0),
    ),
}


class ToolExecutionStoreError(RuntimeError):
    pass


class ToolBindingConflict(ToolExecutionStoreError):
    pass


class ToolExecutionWaitExpired(ToolExecutionStoreError):
    pass


class ClaimState(StrEnum):
    OWNER = "owner"
    WAIT = "wait"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ExecutionClaim:
    state: ClaimState
    result: ToolResult | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not ClaimState or ((self.state is ClaimState.COMPLETED) != (self.result is not None)):
            raise ValueError("Execution claim is invalid")


class ToolExecutionStore(Protocol):
    async def claim(self, request: ToolRequest) -> ExecutionClaim: ...
    async def wait_completed(self, request: ToolRequest) -> ToolResult: ...
    async def complete(self, request: ToolRequest, result: ToolResult) -> None: ...


@dataclass(slots=True)
class InMemoryToolExecutionStore:
    """Development-only execution store."""

    _records: dict[tuple[str, str, str], tuple[str, ToolResult | None]] = field(init=False, default_factory=dict)
    _condition: asyncio.Condition = field(init=False, default_factory=asyncio.Condition)

    async def claim(self, request: ToolRequest) -> ExecutionClaim:
        identity = _identity(request)
        async with self._condition:
            current = self._records.get(identity)
            if current is None:
                self._records[identity] = (request.idempotency_key, None)
                return ExecutionClaim(ClaimState.OWNER)
            if current[0] != request.idempotency_key:
                raise ToolBindingConflict("Tool call identity is already bound")
            return ExecutionClaim(ClaimState.WAIT) if current[1] is None else ExecutionClaim(ClaimState.COMPLETED, snapshot_tool_result(current[1]))

    async def wait_completed(self, request: ToolRequest) -> ToolResult:
        identity = _identity(request)
        async with self._condition:
            while True:
                current = self._records.get(identity)
                if current is None or current[0] != request.idempotency_key:
                    raise ToolBindingConflict("Tool execution claim is missing or changed")
                if current[1] is not None:
                    return snapshot_tool_result(current[1])
                await self._condition.wait()

    async def complete(self, request: ToolRequest, result: ToolResult) -> None:
        result = _validated_result(request, result)
        identity = _identity(request)
        async with self._condition:
            current = self._records.get(identity)
            if current is None or current[0] != request.idempotency_key:
                raise ToolBindingConflict("Tool execution claim is missing or changed")
            if current[1] is not None and current[1] != result:
                raise ToolExecutionStoreError("Tool execution already completed differently")
            self._records[identity] = (request.idempotency_key, result)
            self._condition.notify_all()


class SQLiteToolExecutionStore:
    """Durable owner/wait/completed claims; recovery is an explicit exclusive phase."""

    def __init__(self, path: Path, *, sqlite_busy_timeout_seconds: float = 5.0, poll_interval_seconds: float = 0.05, execution_wait_seconds: float | None = None) -> None:
        if type(path) is not type(Path()) or not path.is_absolute():
            raise ValueError("Tool execution store requires an absolute SQLite path")
        self._path = Path(path)
        self._busy_timeout = _duration(sqlite_busy_timeout_seconds, "SQLite busy timeout")
        self._poll_interval = _duration(poll_interval_seconds, "Execution poll interval")
        self._execution_wait = None if execution_wait_seconds is None else _duration(execution_wait_seconds, "Execution wait")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=self._busy_timeout)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout * 1000)}")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN EXCLUSIVE")
            objects = connection.execute("SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0 and not objects:
                connection.execute(_DDL_META)
                connection.execute(_DDL_EXECUTIONS)
                connection.execute("INSERT INTO schema_meta VALUES (1, ?, ?)", (_VERSION, _FINGERPRINT))
                connection.execute(f"PRAGMA user_version = {_VERSION}")
            else:
                self._validate_schema(connection, version, objects)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection, version: int, objects: list[tuple[str, str, str | None]]) -> None:
        if version != _VERSION or {(kind, name) for kind, name, _ in objects} != {("table", "executions"), ("table", "schema_meta")}:
            raise ToolExecutionStoreError("Unknown tool execution store schema")
        expected_sql = {"schema_meta": _DDL_META, "executions": _DDL_EXECUTIONS}
        actual = {name: sql for _kind, name, sql in objects}
        for name, expected in expected_sql.items():
            if _normalize_sql(actual.get(name)) != _normalize_sql(expected):
                raise ToolExecutionStoreError("Corrupt tool execution store schema")
            columns = tuple((row[1], row[2], row[3], row[5]) for row in connection.execute(f"PRAGMA table_info({name})"))
            if columns != _EXPECTED_COLUMNS[name]:
                raise ToolExecutionStoreError("Corrupt tool execution store schema")
        if connection.execute("SELECT singleton, version, fingerprint FROM schema_meta").fetchall() != [(1, _VERSION, _FINGERPRINT)]:
            raise ToolExecutionStoreError("Corrupt tool execution store metadata")

    async def claim(self, request: ToolRequest) -> ExecutionClaim:
        return await asyncio.to_thread(self._claim, request)

    def _claim(self, request: ToolRequest) -> ExecutionClaim:
        identity = _identity(request)
        row = self._read_claim(identity)
        if row is not None:
            return _claim_from_row(row, request.idempotency_key)
        deadline = time.monotonic() + self._busy_timeout
        while True:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    row = connection.execute(
                        "SELECT binding_key, state, result_json FROM executions "
                        "WHERE session_id=? AND run_id=? AND tool_call_id=?",
                        identity,
                    ).fetchone()
                    if row is not None:
                        return _claim_from_row(row, request.idempotency_key)
                    connection.execute(
                        "INSERT INTO executions VALUES (?, ?, ?, ?, 'started', NULL)",
                        (*identity, request.idempotency_key),
                    )
                    return ExecutionClaim(ClaimState.OWNER)
            except sqlite3.IntegrityError:
                row = self._read_claim(identity)
                if row is not None:
                    return _claim_from_row(row, request.idempotency_key)
            except sqlite3.OperationalError as error:
                if not _is_busy(error):
                    raise ToolExecutionStoreError("Tool execution store operation failed") from None
            if time.monotonic() >= deadline:
                # A competing first claimant may have committed at the busy
                # deadline.  Observe that durable winner before surfacing the
                # stable store-busy error.
                row = self._read_claim(identity)
                if row is not None:
                    return _claim_from_row(row, request.idempotency_key)
                raise ToolExecutionStoreError("Tool execution store is busy") from None
            time.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))

    def _read_claim(self, identity: tuple[str, str, str]) -> tuple[object, ...] | None:
        try:
            with self._connect() as connection:
                return connection.execute(
                    "SELECT binding_key, state, result_json FROM executions "
                    "WHERE session_id=? AND run_id=? AND tool_call_id=?",
                    identity,
                ).fetchone()
        except sqlite3.OperationalError:
            raise ToolExecutionStoreError("Tool execution store read failed") from None

    async def wait_completed(self, request: ToolRequest) -> ToolResult:
        loop = asyncio.get_running_loop()
        deadline = None if self._execution_wait is None else loop.time() + self._execution_wait
        while True:
            claim = await self.claim(request)
            if claim.state is ClaimState.COMPLETED:
                assert claim.result is not None
                return claim.result
            if claim.state is ClaimState.OWNER:
                raise ToolExecutionStoreError("Execution ownership changed while waiting")
            if deadline is not None and loop.time() >= deadline:
                raise ToolExecutionWaitExpired("Tool execution is still owned")
            delay = self._poll_interval if deadline is None else min(self._poll_interval, max(0.0, deadline - loop.time()))
            await asyncio.sleep(delay)

    async def complete(self, request: ToolRequest, result: ToolResult) -> None:
        result = _validated_result(request, result)
        await asyncio.to_thread(self._complete, request, _encode_result(result))

    def _complete(self, request: ToolRequest, encoded: str) -> None:
        identity = _identity(request)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT binding_key, state, result_json FROM executions WHERE session_id=? AND run_id=? AND tool_call_id=?", identity).fetchone()
            if row is None or row[0] != request.idempotency_key:
                raise ToolBindingConflict("Tool execution claim is missing or changed")
            if row[1] == "completed":
                if row[2] != encoded:
                    raise ToolExecutionStoreError("Tool execution already completed differently")
                return
            if row[1] != "started" or row[2] is not None:
                raise ToolExecutionStoreError("Corrupt tool execution record")
            connection.execute("UPDATE executions SET state='completed', result_json=? WHERE session_id=? AND run_id=? AND tool_call_id=?", (encoded, *identity))

    async def recover_abandoned(self) -> int:
        """Settle started rows only when the caller has excluded every live owner."""
        return await asyncio.to_thread(self._recover_abandoned)

    def _recover_abandoned(self) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN EXCLUSIVE")
            rows = connection.execute("SELECT session_id, run_id, tool_call_id, binding_key FROM executions WHERE state='started' ORDER BY session_id, run_id, tool_call_id").fetchall()
            for session_id, run_id, call_id, binding in rows:
                connection.execute("UPDATE executions SET state='completed', result_json=? WHERE session_id=? AND run_id=? AND tool_call_id=? AND state='started'", (_encode_result(_interrupted(binding)), session_id, run_id, call_id))
            connection.commit()
            return len(rows)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


def _duration(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{label} is invalid")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or not 0.001 <= result <= 3600:
        raise ValueError(f"{label} is invalid")
    return result


def _normalize_sql(value: str | None) -> str:
    return "" if type(value) is not str else " ".join(value.replace("\n", " ").split()).strip().lower()


def _is_busy(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return type(code) is int and code & 0xFF in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }


def _claim_from_row(row: tuple[object, ...], binding_key: str) -> ExecutionClaim:
    if len(row) != 3:
        raise ToolExecutionStoreError("Corrupt tool execution record")
    binding, state, encoded = row
    if binding != binding_key:
        raise ToolBindingConflict("Tool call identity is already bound")
    if state == "started" and encoded is None:
        return ExecutionClaim(ClaimState.WAIT)
    if state != "completed" or type(encoded) is not str:
        raise ToolExecutionStoreError("Corrupt tool execution record")
    return ExecutionClaim(ClaimState.COMPLETED, _decode_result(encoded, binding_key))


def _identity(request: ToolRequest) -> tuple[str, str, str]:
    return request.session_id.value, request.run_id.value, request.tool_call_id


def _validated_result(request: ToolRequest, result: ToolResult) -> ToolResult:
    result = snapshot_tool_result(result)
    if result.execution_key != request.idempotency_key:
        raise ToolExecutionStoreError("Tool result is not bound to its execution claim")
    return result


def _interrupted(binding: str) -> ToolResult:
    return ToolResult(ToolResultStatus.FAILED, failure=RunFailure(RunErrorCode.TOOL_FAILED), execution_key=binding)


def _encode_result(result: ToolResult) -> str:
    inline = result.inline_content
    payload = {
        "artifacts": [{"artifact_id": ref.artifact_id, "media_type": ref.media_type, "metadata": ref.metadata, "name": ref.name, "sha256": ref.sha256, "size_bytes": ref.size_bytes} for ref in result.artifacts],
        "execution_key": result.execution_key,
        "failure": result.failure.code.value if result.failure is not None else None,
        "inline": None if inline is None else {"media_type": inline.media_type, "value": inline.value},
        "status": result.status.value,
        "summary": result.summary,
    }
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        raise ToolExecutionStoreError("Tool execution result cannot be encoded") from None
    if len(encoded.encode("utf-8")) > _MAX_RESULT_BYTES:
        raise ToolExecutionStoreError("Tool execution result exceeds its storage limit")
    return encoded


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _constant(_value: str) -> NoReturn:
    raise ValueError


def _check_depth(value: object, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError
    if type(value) is dict:
        for item in value.values():
            _check_depth(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _check_depth(item, depth + 1)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError


def _decode_result(encoded: str, binding: str) -> ToolResult:
    try:
        if type(encoded) is not str or not 0 < len(encoded.encode("utf-8")) <= _MAX_RESULT_BYTES:
            raise ValueError
        value = json.loads(encoded, object_pairs_hook=_pairs, parse_constant=_constant)
        _check_depth(value)
        required = {"artifacts", "execution_key", "failure", "inline", "status", "summary"}
        if type(value) is not dict or set(value) != required or value["execution_key"] != binding:
            raise ValueError
        failure_raw = value["failure"]
        failure = None if failure_raw is None else RunFailure(RunErrorCode(cast(str, failure_raw)))
        inline_raw = value["inline"]
        inline = None
        if inline_raw is not None:
            if type(inline_raw) is not dict or set(inline_raw) != {"media_type", "value"}:
                raise ValueError
            inline = InlineContent(inline_raw["value"], cast(str, inline_raw["media_type"]))
        artifact_values = value["artifacts"]
        if type(artifact_values) is not list or len(artifact_values) > 16:
            raise ValueError
        artifacts = tuple(_decode_ref(item) for item in artifact_values)
        result = ToolResult(ToolResultStatus(cast(str, value["status"])), inline_content=inline, artifacts=artifacts, summary=cast(str | None, value["summary"]), failure=failure, execution_key=binding)
        if _encode_result(result) != encoded:
            raise ValueError
        return result
    except (KeyError, OverflowError, TypeError, ValueError, json.JSONDecodeError):
        raise ToolExecutionStoreError("Corrupt tool execution result") from None


def _decode_ref(value: object) -> ArtifactRef:
    keys = {"artifact_id", "media_type", "metadata", "name", "sha256", "size_bytes"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError
    return ArtifactRef(cast(str, value["artifact_id"]), cast(str, value["media_type"]), cast(int, value["size_bytes"]), cast(str, value["sha256"]), cast(str | None, value["name"]), cast(dict, value["metadata"]))
