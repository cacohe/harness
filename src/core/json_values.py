import json
from dataclasses import dataclass
from typing import NoReturn

MAX_TOOL_ARGUMENT_BYTES = 16_384
_MAX_JSON_DEPTH = 16


@dataclass(frozen=True, slots=True)
class _JsonNumberToken:
    """A number lexeme validated by the JSON scanner."""

    source: str


def validate_bounded_text(text: str, max_bytes: int) -> None:
    if type(text) is not str or len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
        raise ValueError("Expected an exact UTF-8 string within its byte limit")


def validate_json_text(text: str, max_bytes: int, *, allow_boolean_root: bool = False) -> None:
    """Validate strict JSON without rewriting numeric tokens or other source text."""
    try:
        validate_bounded_text(text, max_bytes)
        value = json.loads(
            text,
            parse_float=_JsonNumberToken,
            parse_int=_JsonNumberToken,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        if type(value) is not dict and not (allow_boolean_root and type(value) is bool):
            raise ValueError
        _validate_structure(value, depth=0)
    except (RecursionError, ValueError):
        raise ValueError("Expected bounded strict JSON text") from None


def _reject_constant(_value: str) -> NoReturn:
    raise ValueError


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError
    return result


def _validate_structure(value: object, depth: int) -> None:
    if type(value) is dict or type(value) is list:
        if depth >= _MAX_JSON_DEPTH:
            raise ValueError
        if type(value) is dict:
            for key in value:
                key.encode("utf-8")
        for item in value.values() if type(value) is dict else value:
            _validate_structure(item, depth + 1)
    elif type(value) is str:
        value.encode("utf-8")
    elif value is not None and type(value) not in (bool, _JsonNumberToken):
        raise ValueError
