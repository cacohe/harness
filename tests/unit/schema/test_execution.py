from datetime import datetime, timezone
from math import inf, nan

import pytest
from pydantic import ValidationError
from src.core.identifiers import (
    ExecutionIdFactory,
    UuidExecutionIds,
    validate_execution_id,
)
from src.schema.execution import (
    AgentProfile,
    PendingSessionEvent,
    SessionEvent,
    SessionEventType,
    SessionOutcome,
    SubmissionRecord,
    SubmissionState,
)


def _pending(**updates) -> PendingSessionEvent:
    values = {
        "event_id": "event-1",
        "session_id": "session-1",
        "type": SessionEventType.TURN_STARTED,
        "task_id": "task-1",
        "execution_id": "execution-1",
        "turn_id": "turn-1",
        "payload": {"outcome": SessionOutcome.COMPLETED.value, "nested": [1, True, None]},
    }
    values.update(updates)
    return PendingSessionEvent(**values)


def test_pending_event_is_frozen_strict_json_with_a_utc_timestamp():
    event = _pending()

    assert event.occurred_at.tzinfo is not None
    assert event.occurred_at.utcoffset() == timezone.utc.utcoffset(event.occurred_at)
    assert event.payload == {"outcome": "completed", "nested": [1, True, None]}
    with pytest.raises(ValidationError, match="frozen"):
        event.task_id = "changed"


@pytest.mark.parametrize(
    "payload",
    [
        {"value": nan},
        {"value": inf},
        {"value": -inf},
        {1: "non-string key"},
        {"value": ("tuple",)},
        {"value": SessionOutcome.COMPLETED},
    ],
)
def test_pending_event_rejects_values_that_are_not_exact_json(payload):
    with pytest.raises(ValidationError, match="strict JSON"):
        _pending(payload=payload)


def test_session_event_requires_a_positive_sequence_and_forbids_extra_fields():
    values = _pending().model_dump()

    with pytest.raises(ValidationError):
        SessionEvent(sequence=0, **values)
    with pytest.raises(ValidationError, match="extra"):
        SessionEvent(sequence=1, unexpected=True, **values)

    event = SessionEvent(sequence=1, **values)
    assert event.sequence == 1


@pytest.mark.parametrize("value", ["", "line\nbreak", "null\x00byte", "x" * 257])
def test_execution_ids_reject_empty_control_or_oversized_values(value):
    with pytest.raises(ValueError, match="event_id"):
        validate_execution_id(value, "event_id")


def test_execution_id_limit_is_measured_in_utf8_bytes():
    assert validate_execution_id("界" * 85, "session_id") == "界" * 85
    with pytest.raises(ValueError, match="session_id"):
        validate_execution_id("界" * 86, "session_id")


def test_event_hierarchy_rejects_child_ids_without_their_parent():
    with pytest.raises(ValidationError, match="step_id requires turn_id"):
        _pending(turn_id=None, step_id="step-1")
    with pytest.raises(ValidationError, match="attempt_id requires step_id"):
        _pending(step_id=None, attempt_id="attempt-1")
    with pytest.raises(ValidationError, match="tool_call_id requires step_id"):
        _pending(step_id=None, tool_call_id="tool-1")


def test_submission_record_is_frozen_and_validates_digest_protocol():
    now = datetime.now(timezone.utc)
    record = SubmissionRecord(
        session_id="session-1",
        message_id="message-1",
        digest_algorithm="sha256-raw-utf8-v1",
        message_digest="a" * 64,
        task_id="task-1",
        turn_id="turn-1",
        state=SubmissionState.ACCEPTED,
        created_at=now,
        updated_at=now,
    )

    assert record.state is SubmissionState.ACCEPTED
    with pytest.raises(ValidationError, match="64 lowercase hexadecimal"):
        record.model_copy(update={"message_digest": "not-a-digest"}).model_validate(
            {**record.model_dump(), "message_digest": "not-a-digest"}
        )
    with pytest.raises(ValidationError, match="frozen"):
        record.state = SubmissionState.COMPLETED


def test_agent_profile_forbids_credentials_and_is_frozen():
    profile = AgentProfile(
        id="default-agent",
        version="1",
        system_prompt_digest="1" * 64,
        model_identifier="openai/gpt-4o",
        model_settings_digest="2" * 64,
        skills_digest="3" * 64,
        tools_digest="4" * 64,
        memory_policy_version="none-v1",
        fact_schema_version=1,
    )

    assert "api_key" not in profile.model_dump()
    with pytest.raises(ValidationError, match="extra"):
        AgentProfile(**profile.model_dump(), api_key="secret")
    with pytest.raises(ValidationError, match="frozen"):
        profile.version = "2"


def test_uuid_factory_produces_distinct_valid_ids_for_every_scope():
    factory = UuidExecutionIds()
    generated = {
        factory.session(),
        factory.execution(),
        factory.turn(),
        factory.step(),
        factory.attempt(),
        factory.tool_call(),
        factory.event(),
        factory.message(),
    }

    assert len(generated) == 8
    assert all(validate_execution_id(value, "generated") == value for value in generated)


def test_execution_id_factory_accepts_a_deterministic_structural_fake():
    class SequentialIds:
        def __init__(self) -> None:
            self.next = 0

        def _value(self, prefix: str) -> str:
            self.next += 1
            return f"{prefix}-{self.next}"

        def session(self) -> str:
            return self._value("session")

        def execution(self) -> str:
            return self._value("execution")

        def turn(self) -> str:
            return self._value("turn")

        def step(self) -> str:
            return self._value("step")

        def attempt(self) -> str:
            return self._value("attempt")

        def tool_call(self) -> str:
            return self._value("tool")

        def event(self) -> str:
            return self._value("event")

        def message(self) -> str:
            return self._value("message")

    fake = SequentialIds()

    assert isinstance(fake, ExecutionIdFactory)
    assert [fake.session(), fake.session()] == ["session-1", "session-2"]
