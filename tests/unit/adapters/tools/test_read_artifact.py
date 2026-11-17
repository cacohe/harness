import asyncio
import hashlib

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.tools.read_artifact import ReadArtifactTool
from src.core.artifacts import ArtifactWrite


async def _stream(payload: bytes):
    yield payload[:3]
    yield payload[3:]


def test_reads_only_bounded_slice() -> None:
    async def scenario():
        payload = b"abcdefghij"
        digest = hashlib.sha256(payload).hexdigest()
        repo = InMemoryArtifactRepository(1024)
        ref = await repo.put(
            ArtifactWrite(
                _stream(payload), "text/plain", expected_size_bytes=len(payload),
                expected_sha256=digest, max_bytes=1024,
            )
        )
        result = await ReadArtifactTool(repo, max_read_bytes=4)(
            {"artifact_id": ref.artifact_id, "offset": 2, "limit": 4}
        )
        assert result["content"] == "cdef"
        assert result["returned_bytes"] == 4
        assert result["truncated"] is True

    asyncio.run(scenario())


def test_binary_is_base64_and_reads_only_limit_plus_one() -> None:
    async def scenario():
        payload = b"\x00\x01\x02\x03\x04\x05"
        repo = InMemoryArtifactRepository(1024)
        ref = await repo.put(ArtifactWrite(_stream(payload), "application/octet-stream", max_bytes=1024))
        result = await ReadArtifactTool(repo, max_read_bytes=3)(
            {"artifact_id": ref.artifact_id, "offset": 1, "limit": 3}
        )
        assert result["encoding"] == "base64"
        assert result["content"] == "AQID"
        assert result["returned_bytes"] == 3 and result["truncated"] is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {"artifact_id": "bad", "offset": 0, "limit": 1},
        {"artifact_id": "sha256:" + "0" * 64, "offset": True, "limit": 1},
        {"artifact_id": "sha256:" + "0" * 64, "offset": -1, "limit": 1},
        {"artifact_id": "sha256:" + "0" * 64, "offset": 0, "limit": 5},
        {"artifact_id": "sha256:" + "0" * 64, "offset": 0, "limit": 1, "path": "x"},
    ],
)
def test_rejects_invalid_ranges_and_extra_fields(arguments) -> None:
    with pytest.raises((ValueError, KeyError)):
        asyncio.run(ReadArtifactTool(InMemoryArtifactRepository(1024), max_read_bytes=4)(arguments))


def test_missing_and_invalid_utf8_are_stable() -> None:
    async def scenario():
        repo = InMemoryArtifactRepository(1024)
        tool = ReadArtifactTool(repo)
        with pytest.raises(KeyError):
            await tool({"artifact_id": "sha256:" + "0" * 64, "offset": 0, "limit": 1})
        ref = await repo.put(ArtifactWrite(_stream(b"\xff"), "text/plain", max_bytes=1024))
        with pytest.raises(UnicodeDecodeError):
            await tool({"artifact_id": ref.artifact_id, "offset": 0, "limit": 1})

    asyncio.run(scenario())


def test_untrusted_stat_and_stream_are_rejected() -> None:
    class BadRepository:
        async def stat(self, _artifact_id):
            return object()

        async def delete(self, _artifact_id):
            return None

        async def put(self, _request):
            raise NotImplementedError

        async def open(self, _artifact_id):
            yield b"x"

    with pytest.raises(ValueError):
        asyncio.run(ReadArtifactTool(BadRepository())({"artifact_id": "sha256:" + "0" * 64, "offset": 0, "limit": 1}))
