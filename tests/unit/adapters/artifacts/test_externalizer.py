import asyncio
import hashlib
import json

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.core.artifacts import ArtifactWrite

ALL_TYPES = frozenset({"text/plain", "application/json", "application/octet-stream"})


def _externalizer(repo, *, inline=6, artifact=256, summary=48, allowed=ALL_TYPES):
    return PolicyArtifactExternalizer(repo, ArtifactPolicy(inline, artifact, summary, allowed))


async def _read(repo, artifact_id):
    return b"".join([chunk async for chunk in repo.open(artifact_id)])


@pytest.mark.parametrize(
    "exact,over,media_type,encoded",
    [
        ("界界", "界界x", "text/plain", "界界x".encode()),
        (
            b"\0\x01\x02\x03\x04\x05",
            b"\0\x01\x02\x03\x04\x05\xff",
            "application/octet-stream",
            b"\0\x01\x02\x03\x04\x05\xff",
        ),
        ([1234], [12345], "application/json", b"[12345]"),
    ],
)
def test_inline_threshold_is_exact_encoded_byte_count(exact, over, media_type, encoded):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo)
        inline = await externalizer.externalize(exact, media_type)
        assert inline.inline_content.value == exact
        assert inline.inline_content.size_bytes == 6
        assert inline.artifacts == () and inline.summary is None

        external = await externalizer.externalize(over, media_type)
        assert external.inline_content is None
        assert len(external.artifacts) == 1
        ref = external.artifacts[0]
        assert ref.media_type == media_type
        assert ref.size_bytes == 7
        assert ref.sha256 == hashlib.sha256(encoded).hexdigest()
        assert await repo.stat(ref.artifact_id) == ref
        assert await _read(repo, ref.artifact_id) == encoded

    asyncio.run(scenario())


def test_json_content_is_deterministic_and_summary_retains_keys_and_count():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo, inline=0)
        content = {"z": [1, 2], "alpha": "保密内容"}
        expected = '{"alpha":"保密内容","z":[1,2]}'.encode()
        result = await externalizer.externalize(content, "application/json", "example.json")
        reordered = await externalizer.externalize(
            {"alpha": "保密内容", "z": [1, 2]},
            "application/json",
            "example.json",
        )

        assert result.artifacts[0].artifact_id != reordered.artifacts[0].artifact_id
        assert result.artifacts[0].sha256 == reordered.artifacts[0].sha256
        assert result.artifacts[0].size_bytes == reordered.artifacts[0].size_bytes
        assert result.summary == reordered.summary
        assert result.artifacts[0].name == "example.json"
        assert await _read(repo, result.artifacts[0].artifact_id) == expected
        assert await _read(repo, reordered.artifacts[0].artifact_id) == expected
        assert "2 keys" in result.summary
        assert '"alpha"' in result.summary and '"z"' in result.summary
        assert "保密内容" not in result.summary
        assert len(result.summary.encode()) <= 48

    asyncio.run(scenario())


def test_json_array_summary_keeps_item_count_and_json_null_stays_inline():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo, inline=4)
        inline = await externalizer.externalize(None, "application/json")
        assert inline.inline_content.value is None
        assert inline.inline_content.size_bytes == 4
        external = await externalizer.externalize([1, 2, 3], "application/json")
        assert "3 items" in external.summary

    asyncio.run(scenario())


@pytest.mark.parametrize("probe", ["decode", "reencode-key"])
def test_large_json_preview_does_not_parse_payload_or_reencode_large_keys(monkeypatch, probe):
    key = "a" * 200_000
    content = {key: [{"detail": "v" * 80}] * 20_000, "z": "source"}
    real_loads = json.loads
    real_dumps = json.dumps
    full_key_encodes = 0

    def observe_loads(payload, *args, **kwargs):
        if probe == "decode":
            assert len(payload) < 50, "summary decoded an entire JSON payload"
        return real_loads(payload, *args, **kwargs)

    def observe_dumps(value, *args, **kwargs):
        nonlocal full_key_encodes
        if type(value) is str and value == key:
            full_key_encodes += 1
            if probe == "reencode-key":
                assert full_key_encodes == 1, "summary re-encoded an oversized key"
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", observe_loads)
    monkeypatch.setattr(json, "dumps", observe_dumps)

    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=3_000_000)
        externalizer = _externalizer(repo, inline=0, artifact=3_000_000, summary=40)
        result = await externalizer.externalize(content, "application/json")
        assert result.artifacts[0].size_bytes > 2_000_000
        assert result.summary.startswith('JSON object (2 keys): "aaa')
        assert len(result.summary.encode()) <= 40
        assert full_key_encodes == 1

    asyncio.run(scenario())


def test_externalized_text_summary_omits_content_even_when_the_full_text_would_fit():
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo, inline=2, summary=48)
        result = await externalizer.externalize("abcdef", "text/plain")
        assert result.summary == "abc…ef"
        assert await _read(repo, result.artifacts[0].artifact_id) == b"abcdef"

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 9, 10, 11])
def test_text_summary_preserves_utf8_boundaries_and_prefers_head_and_tail(size):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo, inline=0, summary=size)
        result = await externalizer.externalize("猫🙂界一二三四五六🙂尾", "text/plain")
        summary = result.summary
        assert 0 < len(summary.encode("utf-8")) <= size
        assert "\ufffd" not in summary
        if size >= 9:
            assert summary.startswith("猫")
            assert summary.endswith("尾")

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [2, 23, 25, 27])
def test_structured_summary_near_limit_is_utf8_safe(size):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo, inline=0, summary=size)
        result = await externalizer.externalize(
            {"界🙂端点": "do not reveal"},
            "application/json",
        )
        assert 0 < len(result.summary.encode("utf-8")) <= size
        assert "\ufffd" not in result.summary
        if size >= 23:
            assert "1 key" in result.summary
        assert "do not reveal" not in result.summary

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "content,media_type,name",
    [
        ("x", "text/html", None),
        ("x", "text/plain; charset=utf-8", None),
        ("x", "Text/plain", None),
        ("x", "text/plain", "../display.txt"),
        ({"x": float("nan")}, "application/json", None),
        ({1: "value"}, "application/json", None),
        (bytearray(b"x"), "application/octet-stream", None),
    ],
)
def test_content_is_validated_before_inline_return_or_publication(content, media_type, name):
    async def scenario():
        repo = InMemoryArtifactRepository(max_artifact_bytes=256)
        with pytest.raises(ValueError):
            await _externalizer(repo).externalize(content, media_type, name)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "limits",
    [
        (-1, 256, 48, ALL_TYPES),
        (True, 256, 48, ALL_TYPES),
        (7, 6, 48, ALL_TYPES),
        (0, 0, 48, ALL_TYPES),
        (0, 2**63, 48, ALL_TYPES),
        (0, 256, 0, ALL_TYPES),
        (0, 256, 1_048_577, ALL_TYPES),
        (1_048_577, 2_000_000, 48, ALL_TYPES),
        (0, 256, 48, frozenset({"text/*"})),
        (0, 256, 48, {"text/plain"}),
    ],
)
def test_policy_requires_bounded_integer_limits_and_exact_media_types(limits):
    with pytest.raises(ValueError):
        ArtifactPolicy(*limits)


def test_oversized_content_and_repository_rejection_propagate_without_a_ref():
    async def scenario():
        too_small = InMemoryArtifactRepository(max_artifact_bytes=8)
        for repo, artifact_max in ((too_small, 6), (too_small, 10)):
            externalizer = _externalizer(repo, inline=2, artifact=artifact_max)
            with pytest.raises(ValueError):
                await externalizer.externalize("123456789", "text/plain")
            assert repo._objects == {}

    asyncio.run(scenario())


def test_externalizer_forwards_policy_budget_and_waits_for_publication():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class DelayedRepository(InMemoryArtifactRepository):
            async def put(self, request: ArtifactWrite):
                assert request.max_bytes == 10
                started.set()
                await release.wait()
                return await super().put(request)

        repo = DelayedRepository(max_artifact_bytes=256)
        externalizer = _externalizer(repo, inline=2, artifact=10)
        future = asyncio.create_task(externalizer.externalize("0123456789", "text/plain"))
        await started.wait()
        assert not future.done()
        assert repo._objects == {}
        release.set()
        result = await future
        assert await _read(repo, result.artifacts[0].artifact_id) == b"0123456789"

    asyncio.run(scenario())
