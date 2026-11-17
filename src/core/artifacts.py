from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from io import StringIO
from typing import NoReturn, Protocol, TypeAlias, TypeVar, cast
from unicodedata import category

from src.core.events import JsonValue

MAX_ARTIFACT_SIZE_BYTES = (1 << 63) - 1
MAX_INLINE_CONTENT_BYTES = 1_048_576
MAX_SUMMARY_BYTES = 1_048_576
_DEFAULT_MAX_WRITE_BYTES = 64 * 1_048_576
_MAX_METADATA_BYTES = 65_536
_MAX_JSON_DEPTH = 16
_MAX_ARTIFACT_REFS = 16
_MAX_NAME_BYTES = 256
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MEDIA_TOKEN = r"[a-z0-9][a-z0-9!#$&^_.+-]{0,126}"
_MEDIA_TYPE_PATTERN = re.compile(rf"{_MEDIA_TOKEN}/{_MEDIA_TOKEN}")
_METADATA_URL = re.compile(
    r"(?i)(?<![a-z0-9+.-])(?:[a-z][a-z0-9+.-]*://|(?:file|data|mailto|urn):)"
)
_METADATA_PATH = re.compile(
    r"(?i)(?:\A\s*(?:~|\.{1,2})?[/\\]|"
    r"(?<![\w./\\~+-])(?:(?:~|\.{1,2})?[/\\](?=\S|$)|[a-z]:\S))"
)
_CREDENTIAL_TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+")

ArtifactContent: TypeAlias = "JsonValue | bytes"
_JsonRootObserver: TypeAlias = Callable[[Sequence[str] | int], None]


def artifact_id_from_sha256(sha256: str) -> str:
    """Return the stable content address for a complete lowercase SHA-256 digest."""
    if type(sha256) is not str or not _SHA256_PATTERN.fullmatch(sha256):
        raise ValueError("Artifact SHA-256 must be 64 lowercase hexadecimal characters")
    return "sha256:" + sha256


def validate_artifact_id(artifact_id: str) -> None:
    if (
        type(artifact_id) is not str
        or len(artifact_id) != 71
        or not artifact_id.startswith("sha256:")
        or not _SHA256_PATTERN.fullmatch(artifact_id[7:])
    ):
        raise ValueError("Artifact ID must be sha256:<lowercase digest>")


def validate_media_type(media_type: str) -> None:
    if type(media_type) is not str or not _MEDIA_TYPE_PATTERN.fullmatch(media_type):
        raise ValueError("Artifact media type must be a canonical lowercase type/subtype")


def validate_artifact_name(name: str | None) -> None:
    if name is not None and (
        type(name) is not str
        or not _bounded_text(name, _MAX_NAME_BYTES)
        or name.strip() != name
        or name in {".", ".."}
        or any(character in "/\\:" or category(character) in {"Cc", "Cf"} for character in name)
    ):
        raise ValueError("Artifact name must be a bounded display name")


def _bounded_text(value: str, max_bytes: int) -> bool:
    return 0 < len(value) <= max_bytes and len(value.encode("utf-8")) <= max_bytes


def _validate_size(size: int, *, minimum: int = 0) -> None:
    if type(size) is not int or not minimum <= size <= MAX_ARTIFACT_SIZE_BYTES:
        raise ValueError("Artifact size or write limit is outside its supported range")


def _is_credential_field(key: str) -> bool:
    tokens = {token.lower() for token in _CREDENTIAL_TOKEN.findall(key)}
    return (
        bool({"credential", "credentials"} & tokens)
        or ("key" in tokens and bool({"secret", "access"} & tokens))
        or {"session", "token"} <= tokens
        or ("url" in tokens and bool({"signed", "presigned"} & tokens))
    )


def _json_fragments(
    value: object,
    depth: int,
    max_bytes: int,
    *,
    reject_locations: bool,
    observe_root: _JsonRootObserver | None = None,
) -> Iterator[str]:
    value_type = type(value)
    if type(value) is dict or type(value) is list:
        if depth >= _MAX_JSON_DEPTH:
            raise ValueError("Artifact JSON is nested too deeply")
        if len(value) > max_bytes:
            raise ValueError("Artifact JSON exceeds its byte limit")

    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("Artifact JSON object keys must be strings")
        keys = sorted(value)
        yield "{"
        for index, key in enumerate(keys):
            if index:
                yield ","
            yield from _json_fragments(key, depth + 1, max_bytes, reject_locations=reject_locations)
            if reject_locations and _is_credential_field(key):
                raise ValueError("Artifact metadata must not contain credential fields")
            yield ":"
            yield from _json_fragments(
                value[key],
                depth + 1,
                max_bytes,
                reject_locations=reject_locations,
            )
        yield "}"
        if observe_root is not None:
            observe_root(keys)
    elif type(value) is list:
        count = 0
        yield "["
        for count, item in enumerate(value, start=1):
            if count > 1:
                yield ","
            yield from _json_fragments(
                item, depth + 1, max_bytes, reject_locations=reject_locations
            )
        yield "]"
        if observe_root is not None:
            observe_root(count)
    elif type(value) is str:
        if len(value) > max_bytes:
            raise ValueError("Artifact JSON exceeds its byte limit")
        if reject_locations and (_METADATA_URL.search(value) or _METADATA_PATH.search(value)):
            raise ValueError("Artifact metadata must not contain host paths or URLs")
        yield json.dumps(value, ensure_ascii=False)
    elif value is None or value_type is bool or value_type is int or value_type is float:
        if type(value) is int and value.bit_length() > max_bytes * 4:
            raise ValueError("Artifact JSON exceeds its byte limit")
        yield json.dumps(value, allow_nan=False)
    else:
        raise ValueError("Artifact JSON requires exact JSON types")


def _encode_json(
    value: object,
    max_bytes: int,
    *,
    reject_locations: bool = False,
    observe_root: _JsonRootObserver | None = None,
) -> str:
    encoded = StringIO()
    size = 0
    for fragment in _json_fragments(
        value,
        0,
        max_bytes,
        reject_locations=reject_locations,
        observe_root=observe_root,
    ):
        size += len(fragment.encode("utf-8"))
        if size > max_bytes:
            raise ValueError("Artifact JSON exceeds its byte limit")
        encoded.write(fragment)
    return encoded.getvalue()


def _encode_metadata(metadata: dict[str, JsonValue] | None) -> str:
    if metadata is None:
        return "{}"
    if type(metadata) is not dict:
        raise ValueError("Artifact metadata must be a JSON object")
    return _encode_json(metadata, _MAX_METADATA_BYTES, reject_locations=True)


def _unique_json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Artifact JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> NoReturn:
    raise ValueError("Artifact JSON must use finite numbers")


def _decode_json(encoded: str | bytes) -> JsonValue:
    try:
        return cast(
            JsonValue,
            json.loads(
                encoded,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            ),
        )
    except RecursionError:
        raise ValueError("Artifact JSON is nested too deeply") from None


def _decode_metadata(encoded: str) -> dict[str, JsonValue]:
    if type(encoded) is not str or not _bounded_text(encoded, _MAX_METADATA_BYTES):
        raise ValueError("Artifact metadata exceeds its byte limit")
    value = _decode_json(encoded)
    if type(value) is not dict:
        raise ValueError("Artifact metadata must be a JSON object")
    if _encode_metadata(value) != encoded:
        raise ValueError("Artifact metadata must use canonical JSON")
    return value


def _is_json_media_type(media_type: str) -> bool:
    return media_type == "application/json" or media_type.endswith("+json")


def _encode_content(
    content: ArtifactContent,
    media_type: str,
    max_bytes: int,
    *,
    observe_json_root: _JsonRootObserver | None = None,
) -> bytes:
    if _is_json_media_type(media_type):
        return _encode_json(content, max_bytes, observe_root=observe_json_root).encode("utf-8")
    if type(content) is bytes:
        encoded = content
    elif type(content) is str and len(content) <= max_bytes:
        encoded = content.encode("utf-8")
    else:
        raise ValueError("Artifact requires bounded text or bytes for this media type")
    if len(encoded) > max_bytes:
        raise ValueError("Artifact content exceeds its byte limit")
    return encoded


@dataclass(frozen=True, slots=True, init=False)
class ArtifactRef:
    artifact_id: str
    media_type: str
    size_bytes: int
    sha256: str
    name: str | None
    _metadata_json: str = field(repr=False)

    def __init__(
        self,
        artifact_id: str,
        media_type: str,
        size_bytes: int,
        sha256: str,
        name: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        validate_artifact_id(artifact_id)
        if artifact_id != artifact_id_from_sha256(sha256):
            raise ValueError("Artifact ID does not match the SHA-256 digest")
        validate_media_type(media_type)
        validate_artifact_name(name)
        _validate_size(size_bytes)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "size_bytes", size_bytes)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_metadata_json", _encode_metadata(metadata))

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return _decode_metadata(self._metadata_json)


@dataclass(frozen=True, slots=True, init=False)
class ArtifactWrite:
    """A finite single-use stream and its optional integrity expectations."""

    stream: AsyncIterator[bytes] = field(compare=False, repr=False)
    media_type: str
    name: str | None
    expected_size_bytes: int | None
    expected_sha256: str | None
    max_bytes: int
    _metadata_json: str = field(repr=False)

    def __init__(
        self,
        stream: AsyncIterator[bytes],
        media_type: str,
        name: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
        *,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
        max_bytes: int = _DEFAULT_MAX_WRITE_BYTES,
    ) -> None:
        if not isinstance(stream, AsyncIterator):
            raise ValueError("Artifact stream must be an async iterator")
        validate_media_type(media_type)
        validate_artifact_name(name)
        _validate_size(max_bytes, minimum=1)
        if expected_size_bytes is not None:
            _validate_size(expected_size_bytes)
            if expected_size_bytes > max_bytes:
                raise ValueError("Expected artifact size exceeds its write limit")
        if expected_sha256 is not None:
            artifact_id_from_sha256(expected_sha256)
        object.__setattr__(self, "stream", stream)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "expected_size_bytes", expected_size_bytes)
        object.__setattr__(self, "expected_sha256", expected_sha256)
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "_metadata_json", _encode_metadata(metadata))

    @property
    def metadata(self) -> dict[str, JsonValue]:
        return _decode_metadata(self._metadata_json)


@dataclass(frozen=True, slots=True, init=False)
class InlineContent:
    """A small, copy-isolated payload, including an explicit JSON null."""

    media_type: str
    size_bytes: int
    _encoded: bytes = field(repr=False)
    _is_binary: bool = field(repr=False)

    def __init__(self, value: ArtifactContent, media_type: str) -> None:
        validate_media_type(media_type)
        encoded = _encode_content(value, media_type, MAX_INLINE_CONTENT_BYTES)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "size_bytes", len(encoded))
        object.__setattr__(self, "_encoded", encoded)
        object.__setattr__(self, "_is_binary", type(value) is bytes)

    @property
    def value(self) -> ArtifactContent:
        if self._is_binary:
            return self._encoded
        if _is_json_media_type(self.media_type):
            return _decode_json(self._encoded)
        return self._encoded.decode("utf-8")


def _rebuild_inline(content: InlineContent) -> InlineContent:
    try:
        media_type, size_bytes = content.media_type, content.size_bytes
        encoded, is_binary = content._encoded, content._is_binary
        validate_media_type(media_type)
        if (
            type(size_bytes) is not int
            or type(encoded) is not bytes
            or len(encoded) != size_bytes
            or size_bytes > MAX_INLINE_CONTENT_BYTES
            or type(is_binary) is not bool
        ):
            raise ValueError
        if is_binary:
            value: ArtifactContent = encoded
        elif _is_json_media_type(media_type):
            value = _decode_json(encoded)
        else:
            value = encoded.decode("utf-8")
        rebuilt = InlineContent(value, media_type)
        if rebuilt._encoded != encoded:
            raise ValueError
        return rebuilt
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Invalid inline artifact content") from None


def _rebuild_artifact_ref(ref: ArtifactRef) -> ArtifactRef:
    try:
        return ArtifactRef(
            ref.artifact_id,
            ref.media_type,
            ref.size_bytes,
            ref.sha256,
            ref.name,
            ref.metadata,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Invalid artifact reference") from None


_NestedContent = TypeVar("_NestedContent", InlineContent, ArtifactRef)


def _copy_canonical(value: _NestedContent) -> _NestedContent:
    record_type = type(value)
    result = object.__new__(record_type)
    for member in fields(record_type):
        object.__setattr__(result, member.name, object.__getattribute__(value, member.name))
    return result


@dataclass(frozen=True, slots=True, init=False)
class ExternalizedContent:
    _inline_content: InlineContent | None = field(repr=False)
    _artifacts: tuple[ArtifactRef, ...] = field(repr=False)
    summary: str | None

    def __init__(
        self,
        inline_content: InlineContent | None = None,
        artifacts: tuple[ArtifactRef, ...] = (),
        summary: str | None = None,
    ) -> None:
        if (
            type(artifacts) is not tuple
            or len(artifacts) > _MAX_ARTIFACT_REFS
            or any(type(ref) is not ArtifactRef for ref in artifacts)
        ):
            raise ValueError("Externalized content requires a bounded tuple of artifact refs")
        if inline_content is not None:
            if type(inline_content) is not InlineContent or artifacts or summary is not None:
                raise ValueError("Inline content cannot be combined with artifact refs or summary")
        elif (
            not artifacts
            or type(summary) is not str
            or not _bounded_text(summary, MAX_SUMMARY_BYTES)
        ):
            raise ValueError("Externalized artifacts require a bounded non-empty summary")
        object.__setattr__(
            self,
            "_inline_content",
            _rebuild_inline(inline_content) if inline_content is not None else None,
        )
        object.__setattr__(
            self, "_artifacts", tuple(_rebuild_artifact_ref(ref) for ref in artifacts)
        )
        object.__setattr__(self, "summary", summary)

    @property
    def inline_content(self) -> InlineContent | None:
        return _copy_canonical(self._inline_content) if self._inline_content is not None else None

    @property
    def artifacts(self) -> tuple[ArtifactRef, ...]:
        return tuple(_copy_canonical(ref) for ref in self._artifacts)


class ArtifactRepository(Protocol):
    async def put(self, request: ArtifactWrite) -> ArtifactRef:
        """Consume exact bytes and publish only after validation completes.

        The adapter enforces its own upper limit and the request's write limit.
        On return, the artifact is readable. A repeated address with identical
        descriptors deduplicates; differences raise ValueError without mutation.
        """
        ...

    async def stat(self, artifact_id: str) -> ArtifactRef | None:
        """Return the artifact ref, or None for a missing address."""
        ...

    def open(self, artifact_id: str) -> AsyncIterator[bytes]:
        """Return a new byte iterator. Missing addresses raise KeyError on iteration."""
        ...

    async def delete(self, artifact_id: str) -> None:
        """Delete a stored artifact. Missing addresses are ignored."""
        ...


class ArtifactExternalizer(Protocol):
    async def externalize(
        self,
        content: ArtifactContent,
        media_type: str,
        name: str | None = None,
    ) -> ExternalizedContent:
        """Apply the configured policy and return inline content or a readable ref."""
        ...
