from __future__ import annotations

import base64
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar, cast

from src.core.artifacts import ArtifactRef, ExternalizedContent, InlineContent
from src.core.commands import RunLimits, ToolCallRequest
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalEvent,
    JournalSnapshot,
    SubmissionRecord,
    TranscriptEntry,
)
from src.core.state import (
    ApprovalState,
    ModelCallState,
    RunState,
    RunStatus,
    StepState,
    ToolCallState,
)

_MAX_DOCUMENT_BYTES = 16 * 1_048_576
_T = TypeVar("_T")
_RECORDS: dict[str, type[Any]] = {
    cls.__name__: cls
    for cls in (
        JournalCommit,
        JournalSnapshot,
        JournalEvent,
        RunState,
        RunLimits,
        ApprovalState,
        ModelCallState,
        StepState,
        ToolCallState,
        ToolCallRequest,
        SubmissionRecord,
        TranscriptEntry,
        ConversationSummaryCheckpoint,
    )
}
_IDS = {cls.__name__: cls for cls in (EventId, ProfileId, RunId, SessionId, SubmissionId)}
_ENUMS = {cls.__name__: cls for cls in (RunStatus, RunErrorCode, DomainEventType)}


def encode(value: object) -> str:
    document = {"codec": 1, "value": _pack(value)}
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_DOCUMENT_BYTES:
        raise ValueError("SQLite journal document exceeds its limit")
    return encoded


def decode(encoded: object, expected: type[_T]) -> _T:
    if type(encoded) is not str or len(encoded.encode("utf-8")) > _MAX_DOCUMENT_BYTES:
        raise ValueError("SQLite journal document is invalid")
    document = json.loads(
        encoded, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if type(document) is not dict or set(document) != {"codec", "value"} or document["codec"] != 1:
        raise ValueError("SQLite journal codec version is invalid")
    value = _unpack(document["value"])
    if type(value) is not expected or encode(value) != encoded:
        raise ValueError("SQLite journal document is not canonical")
    return cast(_T, value)


def _pack(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"$": "float", "hex": value.hex()}
    if type(value) is bytes:
        return {"$": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if type(value) is tuple:
        return {"$": "tuple", "items": [_pack(item) for item in value]}
    if type(value) is frozenset:
        if any(type(item) is not str for item in value):
            raise ValueError("SQLite journal set is invalid")
        return {"$": "frozenset", "items": sorted(value)}
    if type(value) is datetime:
        if value.tzinfo is not timezone.utc or value.fold != 0:
            raise ValueError("SQLite journal timestamp is invalid")
        return {"$": "datetime", "iso": value.isoformat(timespec="microseconds")}
    name = type(value).__name__
    if name in _IDS and type(value) is _IDS[name]:
        return {"$": name, "value": _pack(object.__getattribute__(value, "value"))}
    if name in _ENUMS and type(value) is _ENUMS[name]:
        return {"$": name, "value": value.value}
    if type(value) is DomainEvent:
        return {
            "$": "DomainEvent",
            "event_id": _pack(value.event_id),
            "type": _pack(value.type),
            "run_id": _pack(value.run_id),
            "session_id": _pack(value.session_id),
            "timestamp": _pack(value.timestamp),
            "payload": object.__getattribute__(value, "_payload_json"),
        }
    if type(value) is ArtifactRef:
        return {
            "$": "ArtifactRef",
            "artifact_id": value.artifact_id,
            "media_type": value.media_type,
            "size_bytes": value.size_bytes,
            "sha256": value.sha256,
            "name": value.name,
            "metadata": object.__getattribute__(value, "_metadata_json"),
        }
    if type(value) is InlineContent:
        return {
            "$": "InlineContent",
            "media_type": value.media_type,
            "encoded": _pack(object.__getattribute__(value, "_encoded")),
            "binary": object.__getattribute__(value, "_is_binary"),
        }
    if type(value) is ExternalizedContent:
        return {
            "$": "ExternalizedContent",
            "inline": _pack(object.__getattribute__(value, "_inline_content")),
            "artifacts": _pack(object.__getattribute__(value, "_artifacts")),
            "summary": value.summary,
        }
    if type(value) is RunFailure:
        return {"$": "RunFailure", "code": _pack(value.code), "message": value.message}
    if name in _RECORDS and type(value) is _RECORDS[name] and is_dataclass(type(value)):
        return {
            "$": name,
            "fields": {
                member.name: _pack(object.__getattribute__(value, member.name))
                for member in fields(cast(Any, type(value)))
                if member.init
            },
        }
    raise ValueError(f"Unsupported SQLite journal value: {name}")


def _unpack(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is not dict or type(value.get("$")) is not str:
        raise ValueError("SQLite journal tagged value is invalid")
    tag = value["$"]
    if tag == "float":
        _keys(value, {"$", "hex"})
        return float.fromhex(_text(value["hex"]))
    if tag == "bytes":
        _keys(value, {"$", "base64"})
        return base64.b64decode(_text(value["base64"]), validate=True)
    if tag == "tuple":
        _keys(value, {"$", "items"})
        return tuple(_unpack(item) for item in _list(value["items"]))
    if tag == "frozenset":
        _keys(value, {"$", "items"})
        items = _list(value["items"])
        if any(type(item) is not str for item in items):
            raise ValueError("SQLite journal set is invalid")
        text_items = cast(list[str], items)
        if text_items != sorted(set(text_items)):
            raise ValueError("SQLite journal set is not canonical")
        return frozenset(text_items)
    if tag == "datetime":
        _keys(value, {"$", "iso"})
        timestamp = datetime.fromisoformat(_text(value["iso"]))
        if timestamp.tzinfo is not timezone.utc or timestamp.fold != 0:
            raise ValueError("SQLite journal timestamp is invalid")
        return timestamp
    if tag in _IDS:
        _keys(value, {"$", "value"})
        return _IDS[tag](_text(_unpack(value["value"])))
    if tag in _ENUMS:
        _keys(value, {"$", "value"})
        return _ENUMS[tag](_text(value["value"]))
    if tag == "DomainEvent":
        _keys(value, {"$", "event_id", "type", "run_id", "session_id", "timestamp", "payload"})
        payload_text = _text(value["payload"])
        payload = json.loads(
            payload_text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if type(payload) is not dict:
            raise ValueError("SQLite event payload is invalid")
        event = DomainEvent(
            cast(EventId, _unpack(value["event_id"])),
            cast(DomainEventType, _unpack(value["type"])),
            cast(RunId, _unpack(value["run_id"])),
            cast(SessionId, _unpack(value["session_id"])),
            cast(datetime, _unpack(value["timestamp"])),
            payload,
        )
        if object.__getattribute__(event, "_payload_json") != payload_text:
            raise ValueError("SQLite event payload is not canonical")
        return event
    if tag == "ArtifactRef":
        _keys(value, {"$", "artifact_id", "media_type", "size_bytes", "sha256", "name", "metadata"})
        metadata_text = _text(value["metadata"])
        metadata = json.loads(
            metadata_text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        if type(metadata) is not dict:
            raise ValueError("SQLite artifact metadata is invalid")
        ref = ArtifactRef(
            _text(value["artifact_id"]),
            _text(value["media_type"]),
            _int(value["size_bytes"]),
            _text(value["sha256"]),
            _optional_text(value["name"]),
            metadata,
        )
        if object.__getattribute__(ref, "_metadata_json") != metadata_text:
            raise ValueError("SQLite artifact metadata is not canonical")
        return ref
    if tag == "InlineContent":
        _keys(value, {"$", "media_type", "encoded", "binary"})
        encoded = _unpack(value["encoded"])
        if type(encoded) is not bytes or type(value["binary"]) is not bool:
            raise ValueError("SQLite inline content is invalid")
        if value["binary"]:
            content: Any = encoded
        elif _text(value["media_type"]) == "application/json" or _text(
            value["media_type"]
        ).endswith("+json"):
            content = json.loads(
                encoded, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        else:
            content = encoded.decode("utf-8")
        return InlineContent(content, _text(value["media_type"]))
    if tag == "ExternalizedContent":
        _keys(value, {"$", "inline", "artifacts", "summary"})
        return ExternalizedContent(
            cast(InlineContent | None, _unpack(value["inline"])),
            cast(tuple[ArtifactRef, ...], _unpack(value["artifacts"])),
            _optional_text(value["summary"]),
        )
    if tag == "RunFailure":
        _keys(value, {"$", "code", "message"})
        failure = RunFailure(cast(RunErrorCode, _unpack(value["code"])))
        if failure.message != _text(value["message"]):
            raise ValueError("SQLite failure message is invalid")
        return failure
    if tag in _RECORDS:
        _keys(value, {"$", "fields"})
        record = _RECORDS[tag]
        raw_fields = value["fields"]
        if type(raw_fields) is not dict:
            raise ValueError("SQLite record fields are invalid")
        expected = {member.name for member in fields(record) if member.init}
        if set(raw_fields) != expected:
            raise ValueError("SQLite record schema is invalid")
        constructor: Any = record
        return constructor(**{name: _unpack(raw_fields[name]) for name in expected})
    raise ValueError("SQLite journal type tag is unknown")


def _keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("SQLite tagged value has unknown fields")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("SQLite journal JSON has duplicate keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("SQLite journal JSON requires finite numbers")


def _text(value: object) -> str:
    if type(value) is not str:
        raise ValueError("SQLite journal text is invalid")
    value.encode("utf-8")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("SQLite journal integer is invalid")
    return value


def _list(value: object) -> list[object]:
    if type(value) is not list:
        raise ValueError("SQLite journal list is invalid")
    return value
