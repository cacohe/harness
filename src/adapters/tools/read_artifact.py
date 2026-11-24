from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

from src.adapters.tools.catalog import ToolCatalogEntry
from src.core.artifacts.contracts import (
    MAX_ARTIFACT_SIZE_BYTES,
    ArtifactRef,
    ArtifactRepository,
    snapshot_artifact_ref,
    validate_artifact_id,
)
from src.core.tools.contracts import ToolSpec

READ_ARTIFACT_TOOL_NAME = "read_artifact"
_READ_ARTIFACT_SCHEMA = (
    '{"additionalProperties":false,"properties":{'
    '"artifact_id":{"type":"string"},"limit":{"type":"integer"},'
    '"offset":{"type":"integer"}},'
    '"required":["artifact_id","offset","limit"],"type":"object"}'
)


def read_artifact_entry(
    repository: ArtifactRepository,
    *,
    workspace_id: str,
    max_read_bytes: int = 262_144,
    max_result_bytes: int = 1_048_576,
) -> ToolCatalogEntry:
    handler = ReadArtifactTool(
        repository,
        max_read_bytes=max_read_bytes,
        max_result_bytes=max_result_bytes,
    )
    spec = ToolSpec(
        READ_ARTIFACT_TOOL_NAME,
        "Read one bounded byte range from an Artifact by ID.",
        _READ_ARTIFACT_SCHEMA,
        requires_approval=False,
        approval_scope="read",
        workspace_id=workspace_id,
    )
    return ToolCatalogEntry(spec, handler, False)


class ReadArtifactTool:
    """Bounded, streaming read capability that never exposes repository locations."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        max_read_bytes: int = 262_144,
        max_result_bytes: int = 1_048_576,
        max_chunk_bytes: int = 1_048_576,
    ) -> None:
        if type(max_read_bytes) is not int or not 1 <= max_read_bytes <= 1_048_576:
            raise ValueError("Artifact read limit is invalid")
        self._repository = repository
        self._max = max_read_bytes
        if type(max_result_bytes) is not int or not 1 <= max_result_bytes <= 1_048_576:
            raise ValueError("Artifact read result limit is invalid")
        self._max_result = max_result_bytes
        if type(max_chunk_bytes) is not int or not 1 <= max_chunk_bytes <= 8 * 1_048_576:
            raise ValueError("Artifact chunk limit is invalid")
        self._max_chunk = max_chunk_bytes

    async def __call__(self, arguments: dict[str, object]) -> dict[str, object]:
        if type(arguments) is not dict or set(arguments) - {"artifact_id", "offset", "limit"}:
            raise ValueError("Artifact read arguments are invalid")
        if set(arguments) != {"artifact_id", "offset", "limit"}:
            raise ValueError("Artifact read arguments are incomplete")
        artifact_id = arguments["artifact_id"]
        offset = arguments["offset"]
        limit = arguments["limit"]
        validate_artifact_id(artifact_id)  # type: ignore[arg-type]
        if (
            type(offset) is not int
            or type(limit) is not int
            or not 0 <= offset <= MAX_ARTIFACT_SIZE_BYTES
            or not 1 <= limit <= self._max
            or offset + limit > MAX_ARTIFACT_SIZE_BYTES
        ):
            raise ValueError("Artifact read range is invalid")
        try:
            raw_ref = await self._repository.stat(artifact_id)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ValueError("Artifact repository stat failed") from None
        if raw_ref is None:
            raise KeyError("Artifact is not available")
        ref = snapshot_artifact_ref(raw_ref)
        if ref.artifact_id != artifact_id:
            raise ValueError("Artifact repository returned an invalid reference")
        needed = min(ref.size_bytes + 1, offset + limit + 1)
        required = min(ref.size_bytes, needed)
        position = 0
        data = bytearray()
        try:
            iterator = self._repository.open(artifact_id)
        except Exception:
            raise ValueError("Artifact repository open failed") from None
        if not isinstance(iterator, AsyncIterator):
            raise ValueError("Artifact repository returned an invalid stream")
        try:
            while position < needed:
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise ValueError("Artifact repository stream failed") from None
                if type(chunk) is not bytes or not chunk or len(chunk) > self._max_chunk:
                    raise ValueError("Artifact repository returned an invalid stream")
                view = memoryview(chunk)
                if position + len(view) > ref.size_bytes:
                    raise ValueError("Artifact stream exceeds its declared size")
                start = max(0, offset - position)
                end = min(len(view), offset + limit + 1 - position)
                if start < end:
                    data.extend(view[start:end])
                position += len(view)
                if position >= needed:
                    break
        finally:
            closer = getattr(iterator, "aclose", None)
            if callable(closer):
                try:
                    await closer()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise ValueError("Artifact repository stream close failed") from None
        if position < required:
            raise ValueError("Artifact stream ended before its declared size")
        payload = bytes(data[:limit])
        result = _read_result(ref, offset, payload)
        if _encoded_size(result) <= self._max_result:
            return result
        low = 0
        high = len(payload) - 1
        best = _read_result(ref, offset, b"")
        if _encoded_size(best) > self._max_result:
            raise ValueError("Artifact read result limit is too small")
        while low <= high:
            middle = (low + high) // 2
            candidate = _read_result(ref, offset, payload[:middle])
            if _encoded_size(candidate) <= self._max_result:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best


def _read_result(ref: ArtifactRef, offset: int, payload: bytes) -> dict[str, object]:
    truncated = offset + len(payload) < ref.size_bytes
    if (
        ref.media_type.startswith("text/")
        or ref.media_type.endswith("+json")
        or ref.media_type == "application/json"
    ):
        try:
            content: str = payload.decode("utf-8", errors="strict")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(payload).decode("ascii")
            encoding = "base64"
    else:
        content = base64.b64encode(payload).decode("ascii")
        encoding = "base64"
    return {
        "artifact": {
            "artifact_id": ref.artifact_id,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
            "sha256": ref.sha256,
            "name": ref.name,
            "metadata": ref.metadata,
        },
        "offset": offset,
        "returned_bytes": len(payload),
        "truncated": truncated,
        "encoding": encoding,
        "content": content,
    }


def _encoded_size(result: dict[str, object]) -> int:
    return len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
