from __future__ import annotations

import base64

from src.core.artifacts import (
    MAX_ARTIFACT_SIZE_BYTES,
    ArtifactRepository,
    snapshot_artifact_ref,
    validate_artifact_id,
)


class ReadArtifactTool:
    """Bounded, streaming read capability that never exposes repository locations."""

    def __init__(self, repository: ArtifactRepository, *, max_read_bytes: int = 262_144) -> None:
        if type(max_read_bytes) is not int or not 1 <= max_read_bytes <= 1_048_576:
            raise ValueError("Artifact read limit is invalid")
        self._repository = repository
        self._max = max_read_bytes

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
        raw_ref = await self._repository.stat(artifact_id)  # type: ignore[arg-type]
        if raw_ref is None:
            raise KeyError("Artifact is not available")
        ref = snapshot_artifact_ref(raw_ref)
        if ref.artifact_id != artifact_id:
            raise ValueError("Artifact repository returned an invalid reference")
        wanted = limit + 1
        skipped = 0
        data = bytearray()
        async for chunk in self._repository.open(artifact_id):
            if type(chunk) is not bytes or not chunk:
                raise ValueError("Artifact repository returned an invalid stream")
            if skipped < offset:
                take = min(len(chunk), offset - skipped)
                skipped += take
                chunk = chunk[take:]
            if chunk and len(data) < wanted:
                data.extend(chunk[: wanted - len(data)])
            if len(data) >= wanted:
                break
        payload = bytes(data[:limit])
        truncated = len(data) > limit or offset + len(payload) < ref.size_bytes
        if ref.media_type.startswith("text/") or ref.media_type.endswith("+json") or ref.media_type == "application/json":
            content: str = payload.decode("utf-8", errors="strict")
            encoding = "utf-8"
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
