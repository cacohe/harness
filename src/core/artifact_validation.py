import re
from unicodedata import category

_MAX_NAME_BYTES = 256
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MEDIA_TOKEN = r"[a-z0-9][a-z0-9!#$&^_.+-]{0,126}"
_MEDIA_TYPE_PATTERN = re.compile(rf"{_MEDIA_TOKEN}/{_MEDIA_TOKEN}")


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
