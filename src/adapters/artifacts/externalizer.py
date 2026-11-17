from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.core.artifacts import (
    MAX_ARTIFACT_SIZE_BYTES,
    MAX_INLINE_CONTENT_BYTES,
    MAX_SUMMARY_BYTES,
    ArtifactContent,
    ArtifactRepository,
    ArtifactWrite,
    ExternalizedContent,
    InlineContent,
    _encode_content,
    _is_json_media_type,
    validate_artifact_name,
    validate_media_type,
)

_WRITE_CHUNK_BYTES = 65_536
_ELLIPSIS = "…"


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    max_inline_bytes: int
    max_artifact_bytes: int
    max_summary_bytes: int
    allowed_media_types: frozenset[str]

    def __post_init__(self) -> None:
        if (
            type(self.max_inline_bytes) is not int
            or type(self.max_artifact_bytes) is not int
            or type(self.max_summary_bytes) is not int
            or not 0 <= self.max_inline_bytes <= MAX_INLINE_CONTENT_BYTES
            or not max(1, self.max_inline_bytes)
            <= self.max_artifact_bytes
            <= MAX_ARTIFACT_SIZE_BYTES
            or not 1 <= self.max_summary_bytes <= MAX_SUMMARY_BYTES
        ):
            raise ValueError("Artifact policy requires bounded byte limits")
        if type(self.allowed_media_types) is not frozenset:
            raise ValueError("Artifact policy requires a frozenset of media types")
        for media_type in self.allowed_media_types:
            validate_media_type(media_type)


class PolicyArtifactExternalizer:
    """Store oversized content and return its ref with a deterministic preview."""

    def __init__(self, repository: ArtifactRepository, policy: ArtifactPolicy) -> None:
        self._repository = repository
        self._policy = policy

    async def externalize(
        self,
        content: ArtifactContent,
        media_type: str,
        name: str | None = None,
    ) -> ExternalizedContent:
        validate_media_type(media_type)
        validate_artifact_name(name)
        policy = self._policy
        if media_type not in policy.allowed_media_types:
            raise ValueError("Artifact media type is not allowed by the policy")

        payload = _encode_content(content, media_type, policy.max_artifact_bytes)
        if len(payload) <= policy.max_inline_bytes:
            value = json.loads(payload) if _is_json_media_type(media_type) else content
            return ExternalizedContent(inline_content=InlineContent(value, media_type))

        summary = _summarize(payload, media_type, type(content) is bytes, policy.max_summary_bytes)
        ref = await self._repository.put(
            ArtifactWrite(
                _chunks(payload),
                media_type,
                name,
                expected_size_bytes=len(payload),
                max_bytes=policy.max_artifact_bytes,
            )
        )
        return ExternalizedContent(artifacts=(ref,), summary=summary)


async def _chunks(payload: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(payload), _WRITE_CHUNK_BYTES):
        yield payload[start : start + _WRITE_CHUNK_BYTES]


def _summarize(payload: bytes, media_type: str, is_binary: bool, limit: int) -> str:
    if _is_json_media_type(media_type):
        value = json.loads(payload)
        if type(value) is dict:
            return _object_summary(value, limit)
        if type(value) is list:
            return _prefix(f"JSON array ({len(value)} items)", limit)
    if is_binary:
        return _prefix(f"{len(payload)} bytes ({media_type})", limit)
    marker_bytes = len(_ELLIPSIS.encode("utf-8"))
    if limit < marker_bytes:
        return "." * limit
    # An externalized preview always omits some of the original text.
    remaining = min(limit - marker_bytes, len(payload) - 1)
    head_size, tail_size = (remaining + 1) // 2, remaining // 2
    head = payload[:head_size].decode("utf-8", errors="ignore")
    tail = payload[len(payload) - tail_size :].decode("utf-8", errors="ignore")
    return head + _ELLIPSIS + tail


def _object_summary(value: dict[str, object], limit: int) -> str:
    count = len(value)
    noun = "key" if count == 1 else "keys"
    label = f"JSON object ({count} {noun})"
    if len(label) >= limit or not value:
        return label[:limit]
    parts = [label]
    remaining = limit - len(label)
    for index, key in enumerate(value):
        separator = ": " if index == 0 else ", "
        part = _prefix(separator + json.dumps(key, ensure_ascii=False), remaining)
        parts.append(part)
        remaining -= len(part.encode("utf-8"))
        if remaining < 4:
            break
    return "".join(parts)


def _prefix(text: str, limit: int) -> str:
    return text[:limit].encode("utf-8")[:limit].decode("utf-8", errors="ignore")
