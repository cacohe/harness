import asyncio
import hashlib
import json

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
                _stream(payload),
                "text/plain",
                expected_size_bytes=len(payload),
                expected_sha256=digest,
                max_bytes=1024,
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
        ref = await repo.put(
            ArtifactWrite(_stream(payload), "application/octet-stream", max_bytes=1024)
        )
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


def test_missing_and_partial_utf8_falls_back_to_base64() -> None:
    async def scenario():
        repo = InMemoryArtifactRepository(1024)
        tool = ReadArtifactTool(repo)
        with pytest.raises(KeyError):
            await tool({"artifact_id": "sha256:" + "0" * 64, "offset": 0, "limit": 1})
        ref = await repo.put(ArtifactWrite(_stream(b"\xff"), "text/plain", max_bytes=1024))
        result = await tool({"artifact_id": ref.artifact_id, "offset": 0, "limit": 1})
        assert result["encoding"] == "base64"

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
        asyncio.run(
            ReadArtifactTool(BadRepository())(
                {"artifact_id": "sha256:" + "0" * 64, "offset": 0, "limit": 1}
            )
        )


@pytest.mark.parametrize("payload", [b"ab", b"abcdef"])
def test_declared_size_mismatch_is_rejected(payload: bytes) -> None:
    from src.core.artifacts import ArtifactRef

    declared = b"abcd"
    digest = hashlib.sha256(declared).hexdigest()
    ref = ArtifactRef("sha256:" + digest, "application/octet-stream", 4, digest)

    class Repository:
        async def stat(self, _artifact_id):
            return ref

        def open(self, _artifact_id):
            async def chunks():
                yield payload

            return chunks()

    with pytest.raises(ValueError):
        asyncio.run(
            ReadArtifactTool(Repository())(
                {"artifact_id": ref.artifact_id, "offset": 0, "limit": 4}
            )
        )


def test_huge_chunk_is_rejected_and_iterator_is_closed() -> None:
    from src.core.artifacts import ArtifactRef

    payload = b"abcd"
    digest = hashlib.sha256(payload).hexdigest()
    ref = ArtifactRef("sha256:" + digest, "application/octet-stream", 4, digest)
    closed = False

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal closed
            if closed:
                raise StopAsyncIteration
            return payload

        async def aclose(self):
            nonlocal closed
            closed = True

    class Repository:
        async def stat(self, _artifact_id):
            return ref

        def open(self, _artifact_id):
            return Stream()

    with pytest.raises(ValueError):
        asyncio.run(
            ReadArtifactTool(Repository(), max_chunk_bytes=3)(
                {"artifact_id": ref.artifact_id, "offset": 0, "limit": 2}
            )
        )
    assert closed


def test_full_binary_read_is_trimmed_to_the_encoded_result_budget() -> None:
    async def scenario():
        payload = bytes(range(256)) * 256
        repo = InMemoryArtifactRepository(100_000)
        ref = await repo.put(
            ArtifactWrite(_stream(payload), "application/octet-stream", max_bytes=100_000)
        )
        result = await ReadArtifactTool(
            repo,
            max_read_bytes=65_536,
            max_result_bytes=65_536,
        )({"artifact_id": ref.artifact_id, "offset": 0, "limit": 65_536})

        encoded = json.dumps(
            result, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        assert len(encoded) <= 65_536
        assert result["returned_bytes"] < 65_536
        assert result["truncated"] is True

    asyncio.run(scenario())
