from __future__ import annotations

import hashlib
import re

_MAX_PROVIDER_NAME = 64
_HASH_LENGTH = 10
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def provider_tool_name(namespace: str, *segments: str) -> str:
    """Build one deterministic provider-safe tool name from normalized segments."""
    normalized = [_normalize_segment(value) for value in (namespace, *segments)]
    full = "__".join(normalized)
    if len(full) <= _MAX_PROVIDER_NAME:
        return full
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    prefix_length = _MAX_PROVIDER_NAME - _HASH_LENGTH - 1
    return f"{full[:prefix_length]}_{digest}"


def _normalize_segment(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("tool name segments must contain provider-safe ASCII characters")
    normalized = _UNSAFE_NAME.sub("_", value).strip("_-")
    if not normalized:
        raise ValueError(f"tool name segment {value!r} has no provider-safe ASCII characters")
    return normalized
