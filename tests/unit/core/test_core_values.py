import json
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
from pathlib import PureWindowsPath

import pytest
from src.core.commands import CancelRun, DecideToolCall, ResumeRun, RunLimits, SubmitRun
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId

ID_TYPES = (RunId, SessionId, SubmissionId, ProfileId, EventId)


@pytest.mark.parametrize("id_type", ID_TYPES)
@pytest.mark.parametrize("value", [None, "", "line\nfeed", "del\x7f", "next\x85", "é" * 129])
def test_ids_reject_invalid_values(id_type, value):
    with pytest.raises(ValueError):
        id_type(value)


@pytest.mark.parametrize("id_type", ID_TYPES)
def test_ids_accept_exact_utf8_limit_and_are_frozen(id_type):
    value = "é" * 128
    identifier = id_type(value)

    assert identifier.value == value
    assert isinstance(hash(identifier), int)
    with pytest.raises(FrozenInstanceError):
        identifier.value = "different"


def test_id_types_do_not_compare_equal_across_scopes():
    assert len({id_type("shared") for id_type in ID_TYPES}) == len(ID_TYPES)


def test_id_validates_the_underlying_utf8_bytes_of_a_str_subclass():
    class MisreportedEncoding(str):
        def encode(self, *args, **kwargs):
            return b"short"

    with pytest.raises(ValueError, match="256 UTF-8 bytes"):
        RunId(MisreportedEncoding("é" * 128 + "a"))


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("field", ["max_steps", "max_elapsed_seconds", "max_inline_result_bytes"])
def test_limits_reject_non_positive_or_non_finite_values(field, value):
    fields = {"max_steps": 1, "max_elapsed_seconds": 0.5, "max_inline_result_bytes": 1}
    fields[field] = value

    with pytest.raises(ValueError):
        RunLimits(**fields)


@pytest.mark.parametrize("field", ["max_steps", "max_inline_result_bytes"])
def test_count_and_byte_limits_require_integers(field):
    fields = {"max_steps": 1, "max_elapsed_seconds": 0.5, "max_inline_result_bytes": 1}
    fields[field] = 1.5

    with pytest.raises(ValueError):
        RunLimits(**fields)


def test_limits_keep_explicit_budget_values():
    limits = RunLimits(1, 0.5, 2)

    assert asdict(limits) == {
        "max_steps": 1,
        "max_elapsed_seconds": 0.5,
        "max_inline_result_bytes": 2,
    }
    with pytest.raises(FrozenInstanceError):
        limits.max_steps = 2


def test_commands_keep_run_and_session_correlation():
    run_id = RunId("run")
    session_id = SessionId("session")
    commands = (
        SubmitRun(run_id, session_id, SubmissionId("submission"), ProfileId("profile"), "hello"),
        ResumeRun(run_id, session_id),
        CancelRun(run_id, session_id),
        DecideToolCall(run_id, session_id, "call", approved=True),
    )

    for command in commands:
        assert command.run_id == run_id
        assert command.session_id == session_id
        with pytest.raises(FrozenInstanceError):
            command.run_id = RunId("other")


def test_failure_serializes_only_stable_public_details():
    failure = RunFailure(RunErrorCode.MODEL_FAILED)

    assert json.loads(json.dumps(asdict(failure))) == {
        "code": "model_failed",
        "message": "Model request failed.",
    }
    with pytest.raises(FrozenInstanceError):
        failure.message = "changed"
    with pytest.raises(ValueError):
        RunFailure(RuntimeError("adapter secret"))


def _event(payload):
    return DomainEvent(
        event_id=EventId("event"),
        type=DomainEventType.RUN_STARTED,
        run_id=RunId("run"),
        session_id=SessionId("session"),
        timestamp=datetime(2026, 11, 17, tzinfo=timezone.utc),
        payload=payload,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"items": [b"binary"]},
        {"nested": {17: "value"}},
        {"nested": [float("nan")]},
        {"value": float("inf")},
        {"value": (1, 2)},
        {"path": PureWindowsPath("C:/private")},
        {"path": "/var/private"},
        {"path": "C:\\private\\file"},
        {"path": "\\\\host\\share\\file"},
        {"path": "file:///var/private"},
        {"nested": {"/var/private": True}},
    ],
)
def test_event_rejects_non_json_values_and_host_paths(payload):
    with pytest.raises(ValueError):
        _event(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "Could not open /var/private/config.json"},
        {"message": "Could not open 'C:\\private\\config.json'"},
        {"message": "Could not open (file:///var/private/config.json)"},
        {"path": "private/config.json"},
        {"source_path": "config.json"},
        {"details": {"sourcePaths": ["private\\config.json"]}},
    ],
)
def test_event_rejects_paths_in_diagnostics_and_explicit_path_fields(payload):
    with pytest.raises(ValueError, match="host paths"):
        _event(payload)


@pytest.mark.parametrize(
    "message",
    [
        "Could not open:/var/private/config.json",
        r"Could not open:C:\private\config.json",
    ],
)
def test_event_rejects_paths_after_a_diagnostic_colon(message):
    with pytest.raises(ValueError, match="host paths"):
        _event({"message": message})


@pytest.mark.parametrize(
    "message",
    [
        "Changed from x:1 to x:2",
        "See:https://example.org/artifact",
    ],
)
def test_event_preserves_non_path_diagnostic_text_with_colons(message):
    assert _event({"message": message}).payload == {"message": message}


@pytest.mark.parametrize(
    "message",
    [
        "https://example.org/releases/v1:/artifact",
        'See "https://example.org/releases/v1:/artifact"',
    ],
)
def test_event_preserves_internal_colons_in_https_urls(message):
    assert _event({"message": message}).payload == {"message": message}


def test_event_rejects_a_host_path_after_a_quoted_https_url():
    message = 'See "https://example.org/releases/v1:/artifact":/var/private/config.json'

    with pytest.raises(ValueError, match="host paths"):
        _event({"message": message})


def test_event_bounds_utf8_payload_size():
    # The compact JSON object framing takes 8 bytes.
    payload = {"v": "é" * ((1_048_576 - 8) // 2)}
    assert _event(payload).payload == payload

    with pytest.raises(ValueError):
        _event({"v": payload["v"] + "é"})


def test_event_bounds_nesting_depth():
    nested = True
    for _ in range(16):
        nested = [nested]

    with pytest.raises(ValueError):
        _event({"nested": nested})


def test_event_rejects_circular_payload():
    payload = {}
    payload["cycle"] = payload

    with pytest.raises(ValueError):
        _event(payload)


def test_event_exposes_a_json_snapshot():
    payload = {"items": [{"value": [True, None, 3, 1.5, "https://example.org/artifact"]}]}
    event = _event(payload)
    payload["items"][0]["value"].append("original changed")
    event.payload["items"][0]["value"].append("copy changed")

    assert event.type.value == "run.started"
    assert event.payload == {
        "items": [{"value": [True, None, 3, 1.5, "https://example.org/artifact"]}]
    }
    assert json.loads(json.dumps(event.payload, allow_nan=False)) == event.payload
    with pytest.raises(FrozenInstanceError):
        event.payload = {}
