from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from io import StringIO
from typing import TypeAlias, cast

from src.core.ids import EventId, RunId, SessionId

_MAX_PAYLOAD_BYTES = 1_048_576
_MAX_PAYLOAD_DEPTH = 16
_WEB_URL_START = re.compile(r"(?<![\w+.-])https?://", re.IGNORECASE)
_WEB_URL_DELIMITERS = frozenset('<>"`\\“”‘’')
# A URI's "://" separator does not introduce a local path.
_HOST_PATH = re.compile(
    r"""^(?:[a-z]:[\\/]|(?:~|\.{1,2})?[\\/]|file:)"""
    r"""|(?:[\s"'`([{=<“‘]|:(?!//))(?:[a-z]:[\\/]|(?:~|\.{1,2})?[\\/]\S|file:)""",
    re.IGNORECASE,
)
_PATH_FIELD = re.compile(r"(?i:(?:^|[_.-])paths?$)|Paths?$")

JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"


class DomainEventType(StrEnum):
    RUN_STARTED = "run.started"
    RUN_WAITING_APPROVAL = "run.waiting_approval"
    RUN_RESUMED = "run.resumed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    TOOL_CALL_DECIDED = "tool.call.decided"


def _text_outside_web_urls(value: str) -> str:
    """Skip URL tokens while retaining their delimiters for host-path checks."""
    fragments: list[str] = []
    cursor = 0
    for match in _WEB_URL_START.finditer(value):
        if match.start() < cursor:
            continue
        # An apostrophe is URL data unless the URL starts inside single quotes.
        single_quoted = match.start() > 0 and value[match.start() - 1] == "'"
        end = match.end()
        while end < len(value):
            character = value[end]
            if character.isspace() or character in _WEB_URL_DELIMITERS:
                break
            if single_quoted and character == "'":
                break
            end += 1
        # Do not exempt a scheme without a host-like first character.
        if end == match.end() or value[match.end()] in "/?#'":
            continue
        fragments.extend((value[cursor : match.start()], " "))
        cursor = end
    fragments.append(value[cursor:])
    return "".join(fragments)


def _json_fragments(value: object, depth: int, *, path_value: bool = False) -> Iterator[str]:
    if isinstance(value, (dict, list)) and depth >= _MAX_PAYLOAD_DEPTH:
        raise ValueError("Event payload is nested too deeply")

    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("Event payload object keys must be strings")
        yield "{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield ","
            yield from _json_fragments(key, depth + 1)
            yield ":"
            yield from _json_fragments(
                value[key], depth + 1, path_value=path_value or bool(_PATH_FIELD.search(key))
            )
        yield "}"
    elif type(value) is list:
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _json_fragments(item, depth + 1, path_value=path_value)
        yield "]"
    elif value is None or type(value) in (bool, int, float, str):
        if isinstance(value, str):
            if len(value) > _MAX_PAYLOAD_BYTES:
                raise ValueError("Event payload exceeds its UTF-8 byte limit")
            if (path_value and value) or _HOST_PATH.search(_text_outside_web_urls(value.lstrip())):
                raise ValueError("Event payload must not contain host paths")
        yield json.dumps(value, allow_nan=False, ensure_ascii=False)
    else:
        raise ValueError("Event payload values must be JSON types")


def _encode_payload(payload: dict[str, JsonValue]) -> str:
    if type(payload) is not dict:
        raise ValueError("Event payload must be a JSON object")
    encoded = StringIO()
    size = 0
    for fragment in _json_fragments(payload, depth=0):
        size += len(fragment.encode("utf-8"))
        if size > _MAX_PAYLOAD_BYTES:
            raise ValueError("Event payload exceeds its UTF-8 byte limit")
        encoded.write(fragment)
    return encoded.getvalue()


@dataclass(frozen=True, init=False)
class DomainEvent:
    event_id: EventId
    type: DomainEventType
    run_id: RunId
    session_id: SessionId
    timestamp: datetime
    _payload_json: str = field(init=False, repr=False)

    def __init__(
        self,
        event_id: EventId,
        type: DomainEventType,
        run_id: RunId,
        session_id: SessionId,
        timestamp: datetime,
        payload: dict[str, JsonValue],
    ) -> None:
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "_payload_json", _encode_payload(payload))

    @property
    def payload(self) -> dict[str, JsonValue]:
        """Return an independent JSON object with the event's payload."""
        return cast(dict[str, JsonValue], json.loads(self._payload_json))
