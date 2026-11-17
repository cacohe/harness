import asyncio
import hashlib

import pytest
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.core.artifacts import ArtifactWrite


def _address(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


async def _chunks(*items: bytes):
    for item in items:
        yield item


async def _read(repo: InMemoryArtifactRepository, artifact_id: str) -> bytes:
    return b"".join([chunk async for chunk in repo.open(artifact_id)])


def test_put_is_streaming_and_open_has_independent_readers():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        artifact_id = _address(b"abc")

        async def stream():
            yield b"a"
            assert await repo.stat(artifact_id) is None
            yield b"bc"

        ref = await repo.put(
            ArtifactWrite(
                stream(),
                "text/plain",
                expected_size_bytes=3,
                expected_sha256=artifact_id[7:],
            )
        )

        assert ref.artifact_id == artifact_id
        assert ref.size_bytes == 3
        assert ref == await repo.stat(artifact_id)
        assert await asyncio.gather(_read(repo, artifact_id), _read(repo, artifact_id)) == [
            b"abc",
            b"abc",
        ]
        await repo.delete(artifact_id)
        await repo.delete(artifact_id)
        assert await repo.stat(artifact_id) is None
        with pytest.raises(KeyError):
            await _read(repo, artifact_id)

    asyncio.run(scenario())


def test_empty_stream_and_equivalent_metadata_deduplicate():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        first = await repo.put(
            ArtifactWrite(
                _chunks(),
                "text/plain",
                name="same",
                metadata={"z": 1, "a": ["same"]},
            )
        )
        duplicate = await repo.put(
            ArtifactWrite(
                _chunks(b""),
                "text/plain",
                name="same",
                metadata={"a": ["same"], "z": 1},
            )
        )

        assert first == duplicate == await repo.stat(_address(b""))
        duplicate.metadata["z"] = 2
        assert (await repo.stat(_address(b""))).metadata == {"z": 1, "a": ["same"]}
        assert await _read(repo, _address(b"")) == b""

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changed",
    [
        {"media_type": "application/octet-stream"},
        {"name": "different"},
        {"metadata": {"version": True}},
    ],
)
def test_concurrent_descriptor_conflict_preserves_first_complete_publication(changed):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        blocked, resume = asyncio.Event(), asyncio.Event()
        descriptor = {"media_type": "text/plain", "name": "original", "metadata": {"version": 1}}

        async def slow():
            blocked.set()
            await resume.wait()
            yield b"abc"

        pending = asyncio.create_task(repo.put(ArtifactWrite(slow(), **(descriptor | changed))))
        await blocked.wait()
        first = await repo.put(ArtifactWrite(_chunks(b"abc"), **descriptor))
        resume.set()
        with pytest.raises(ValueError, match="conflict"):
            await pending

        assert first == await repo.stat(_address(b"abc"))
        assert await _read(repo, first.artifact_id) == b"abc"

    asyncio.run(scenario())


@pytest.mark.parametrize("request_limit,repo_limit", [(2, 10), (10, 2)])
def test_limit_rejects_during_consumption_with_zero_publication(request_limit, repo_limit):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=repo_limit)

        async def oversized():
            yield b"ab"
            yield b"c"
            pytest.fail("oversized stream was read past its first offending chunk")

        with pytest.raises(ValueError):
            await repo.put(ArtifactWrite(oversized(), "text/plain", max_bytes=request_limit))
        assert await repo.stat(_address(b"abc")) is None
        assert await repo.stat(_address(b"ab")) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_size_bytes": 2},
        {"expected_size_bytes": 4},
        {"expected_sha256": "0" * 64},
    ],
)
def test_expectation_mismatch_publishes_nothing(kwargs):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=5)
        with pytest.raises(ValueError):
            await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", **kwargs))
        assert await repo.stat(_address(b"abc")) is None

    asyncio.run(scenario())


class _BytesSubclass(bytes):
    pass


@pytest.mark.parametrize("chunk", [bytearray(b"abc"), memoryview(b"abc"), _BytesSubclass(b"abc")])
def test_chunks_must_be_exact_immutable_bytes(chunk):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        with pytest.raises(ValueError):
            await repo.put(ArtifactWrite(_chunks(chunk), "text/plain"))
        assert await repo.stat(_address(b"abc")) is None

    asyncio.run(scenario())


def test_iterator_failure_publishes_nothing():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)

        async def failed():
            yield b"abc"
            raise RuntimeError("stream failure")

        with pytest.raises(RuntimeError, match="stream failure"):
            await repo.put(ArtifactWrite(failed(), "text/plain"))
        assert await repo.stat(_address(b"abc")) is None

    asyncio.run(scenario())


def test_cancelled_put_publishes_nothing():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        blocked = asyncio.Event()

        async def suspended():
            yield b"abc"
            blocked.set()
            await asyncio.Future()

        put = asyncio.create_task(repo.put(ArtifactWrite(suspended(), "text/plain")))
        await blocked.wait()
        put.cancel()
        with pytest.raises(asyncio.CancelledError):
            await put
        assert await repo.stat(_address(b"abc")) is None

    asyncio.run(scenario())


def test_cancellation_can_interrupt_an_iterator_that_does_not_suspend():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        entered = asyncio.Event()
        consumed = 0

        async def stream():
            nonlocal consumed
            entered.set()
            for _ in range(1_000):
                consumed += 1
                yield b""

        pending = asyncio.create_task(repo.put(ArtifactWrite(stream(), "text/plain")))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert consumed < 1_000
        assert await repo.stat(_address(b"")) is None

    asyncio.run(scenario())


def test_repository_isolates_request_and_receipt_records():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        blocked, resume = asyncio.Event(), asyncio.Event()

        async def stream():
            blocked.set()
            await resume.wait()
            yield b"abc"

        request = ArtifactWrite(stream(), "text/plain", name="initial")
        task = asyncio.create_task(repo.put(request))
        await blocked.wait()
        object.__setattr__(request, "name", "mutated")
        resume.set()
        ref = await task
        assert ref.name == "initial"

        object.__setattr__(ref, "name", "mutated")
        saved = await repo.stat(ref.artifact_id)
        assert saved.name == "initial"
        object.__setattr__(saved, "name", "also mutated")
        assert (await repo.stat(ref.artifact_id)).name == "initial"

    asyncio.run(scenario())


def test_forged_write_is_validated_before_stream_consumption():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)

        async def stream():
            pytest.fail("invalid request consumed its stream")
            yield b"abc"

        request = ArtifactWrite(stream(), "text/plain")
        object.__setattr__(request, "media_type", "text/plain\r\n")
        with pytest.raises(ValueError):
            await repo.put(request)

    asyncio.run(scenario())


def test_forged_metadata_text_rejects_duplicate_keys_before_reading_stream():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)

        async def stream():
            pytest.fail("invalid request consumed its stream")
            yield b"abc"

        request = ArtifactWrite(stream(), "text/plain")
        object.__setattr__(request, "_metadata_json", '{"outer":{"same":1,"same":2}}')
        with pytest.raises(ValueError):
            await repo.put(request)

    asyncio.run(scenario())


@pytest.mark.parametrize("artifact_id", ["../abc", "C:\\abc", "sha256:" + "g" * 64])
def test_repository_validates_id_for_all_lookup_operations(artifact_id):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3)
        with pytest.raises(ValueError):
            await repo.stat(artifact_id)
        with pytest.raises(ValueError):
            await _read(repo, artifact_id)
        with pytest.raises(ValueError):
            await repo.delete(artifact_id)

    asyncio.run(scenario())
