from asyncio import sleep
from collections.abc import AsyncIterator
from hashlib import sha256
from threading import Lock

from src.core.artifacts import (
    MAX_ARTIFACT_SIZE_BYTES,
    ArtifactRef,
    ArtifactWrite,
    artifact_id_from_sha256,
    validate_artifact_id,
)

_READ_CHUNK_BYTES = 65_536
_CHUNKS_PER_CHECKPOINT = 256


class InMemoryArtifactRepository:
    """Stream writes into bounded memory and publish immutable objects atomically."""

    def __init__(self, max_artifact_bytes: int) -> None:
        if (
            type(max_artifact_bytes) is not int
            or not 1 <= max_artifact_bytes <= MAX_ARTIFACT_SIZE_BYTES
        ):
            raise ValueError("Maximum artifact size must be a positive bounded integer")
        self._max_artifact_bytes = max_artifact_bytes
        self._objects: dict[str, tuple[ArtifactRef, bytes]] = {}
        self._lock = Lock()

    async def put(self, request: ArtifactWrite) -> ArtifactRef:
        request = _validated_write(request)
        limit = min(request.max_bytes, self._max_artifact_bytes)
        expected_size = request.expected_size_bytes
        if expected_size is not None:
            if expected_size > limit:
                raise ValueError("Expected artifact size exceeds the repository write limit")
            limit = expected_size

        content = bytearray()
        digest = sha256()
        chunks_since_checkpoint = 0
        async for chunk in request.stream:
            if type(chunk) is not bytes:
                raise ValueError("Artifact stream chunks must be exact bytes")
            if len(content) + len(chunk) > limit:
                raise ValueError("Artifact stream exceeds its write limit or expected size")
            content.extend(chunk)
            digest.update(chunk)
            chunks_since_checkpoint += 1
            if chunks_since_checkpoint == _CHUNKS_PER_CHECKPOINT:
                await sleep(0)
                chunks_since_checkpoint = 0

        checksum = digest.hexdigest()
        if expected_size is not None and expected_size != len(content):
            raise ValueError("Artifact stream size does not match the expected size")
        if request.expected_sha256 is not None and request.expected_sha256 != checksum:
            raise ValueError("Artifact stream SHA-256 does not match the expected digest")
        artifact_id = artifact_id_from_sha256(checksum)
        ref = ArtifactRef(
            artifact_id,
            request.media_type,
            len(content),
            checksum,
            request.name,
            request.metadata,
        )
        receipt = _copy_ref(ref)
        data = bytes(content)
        # An iterator can complete without suspending. Honor pending cancellation before publication.
        await sleep(0)
        with self._lock:
            existing = self._objects.get(artifact_id)
            if existing is not None and existing[0] != ref:
                raise ValueError("Artifact descriptor conflict for content address")
            if existing is None:
                self._objects[artifact_id] = (ref, data)
        return receipt

    async def stat(self, artifact_id: str) -> ArtifactRef | None:
        validate_artifact_id(artifact_id)
        with self._lock:
            entry = self._objects.get(artifact_id)
        return _copy_ref(entry[0]) if entry is not None else None

    async def open(self, artifact_id: str) -> AsyncIterator[bytes]:
        validate_artifact_id(artifact_id)
        with self._lock:
            _, data = self._objects[artifact_id]
        for start in range(0, len(data), _READ_CHUNK_BYTES):
            yield data[start : start + _READ_CHUNK_BYTES]

    async def delete(self, artifact_id: str) -> None:
        validate_artifact_id(artifact_id)
        with self._lock:
            self._objects.pop(artifact_id, None)


def _validated_write(request: ArtifactWrite) -> ArtifactWrite:
    if type(request) is not ArtifactWrite:
        raise ValueError("Expected an ArtifactWrite")
    try:
        return ArtifactWrite(
            request.stream,
            request.media_type,
            request.name,
            request.metadata,
            expected_size_bytes=request.expected_size_bytes,
            expected_sha256=request.expected_sha256,
            max_bytes=request.max_bytes,
        )
    except (AttributeError, TypeError, ValueError, RecursionError):
        raise ValueError("Invalid artifact write request") from None


def _copy_ref(ref: ArtifactRef) -> ArtifactRef:
    return ArtifactRef(
        ref.artifact_id,
        ref.media_type,
        ref.size_bytes,
        ref.sha256,
        ref.name,
        ref.metadata,
    )
