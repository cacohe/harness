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
_QUOTE_END = {"'": "'", '"': '"', "`": "`", "<": ">", "“": "”", "‘": "’"}
# A URI's "://" separator does not introduce a local path.
_HOST_PATH = re.compile(
    r"""^(?:[a-z]:[\\/]|(?:~|\.{1,2})?[\\/]|file:)"""
    r"""|(?:[\s"'`([{=<>“”‘’]|:(?!//))(?:[a-z]:[\\/]|(?:~|\.{1,2})?[\\/]\S|file:)""",
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
    """Mask URL tokens in one forward pass, retaining host-path boundaries."""
    fragments: list[str] = []
    cursor = 0
    retained = 0
    closing_quote = ""
    while cursor < len(value):
        character = value[cursor]
        # match() inspects only the fixed-size scheme at the current position.
        match = _WEB_URL_START.match(value, cursor) if character in "hH" else None
        if match is not None and match.end() < len(value):
            first = value[match.end()]
            # Reject a hostless prefix before scanning any of its suffix.
            if not (first.isspace() or first in _WEB_URL_DELIMITERS or first in "/?#'"):
                fragments.extend((value[retained:cursor], " "))
                cursor = match.end()
                while cursor < len(value):
                    character = value[cursor]
                    if character.isspace() or character in _WEB_URL_DELIMITERS:
                        break
                    if character == "'" and closing_quote == "'":
                        break
                    cursor += 1
                retained = cursor
                continue
        # A word-internal apostrophe is prose, unless it directly quotes a URL.
        if (
            character == "'"
            and 0 < cursor < len(value) - 1
            and value[cursor - 1].isalnum()
            and value[cursor + 1].isalnum()
            and _WEB_URL_START.match(value, cursor + 1) is None
        ):
            cursor += 1
            continue
        if character == closing_quote:
            closing_quote = ""
        elif not closing_quote and character in _QUOTE_END:
            closing_quote = _QUOTE_END[character]
        cursor += 1
    fragments.append(value[retained:])
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
