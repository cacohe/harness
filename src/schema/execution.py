from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_ALGORITHM = "sha256-raw-utf8-v1"
_MAX_ID_BYTES = 256


def validate_execution_id(value: str, name: str) -> str:
    """Return one bounded printable identity accepted by durable schemas."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    if len(value.encode("utf-8")) > _MAX_ID_BYTES:
        raise ValueError(f"{name} must not exceed {_MAX_ID_BYTES} UTF-8 bytes")
    return value


class SessionEventType(str, Enum):
    SESSION_STARTED = "session.started"
    SUBMISSION_ACCEPTED = "submission.accepted"
    EXECUTION_STARTED = "execution.started"
    TURN_STARTED = "turn.started"
    USER_MESSAGE = "user.message"
    STEP_STARTED = "step.started"
    MODEL_ATTEMPT_STARTED = "model.attempt.started"
    MODEL_ATTEMPT_SETTLED = "model.attempt.settled"
    ASSISTANT_SETTLED = "assistant.settled"
    TOOL_CALL_REQUESTED = "tool.call.requested"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_SETTLED = "tool.call.settled"
    STEP_PAUSED = "step.paused"
    STEP_RESUMED = "step.resumed"
    STEP_SETTLED = "step.settled"
    TURN_PAUSED = "turn.paused"
    TURN_RESUMED = "turn.resumed"
    TURN_SETTLED = "turn.settled"
    SUBMISSION_PAUSED = "submission.paused"
    SUBMISSION_RESUMED = "submission.resumed"
    SUBMISSION_SETTLED = "submission.settled"
    EXECUTION_SETTLED = "execution.settled"
    PROFILE_CHANGED = "profile.changed"
    SESSION_INTERRUPTED = "session.interrupted"
    LEGACY_SNAPSHOT_IMPORTED = "legacy.snapshot.imported"


class SessionOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"


class SubmissionState(str, Enum):
    ACCEPTED = "accepted"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


def _strict_json(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("payload must contain strict JSON values")
        return value
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("payload must contain strict JSON values")
        return {key: _strict_json(item) for key, item in value.items()}
    raise ValueError("payload must contain strict JSON values")


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


class PendingSessionEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    event_id: str
    session_id: str
    type: SessionEventType
    task_id: str
    execution_id: str
    turn_id: str | None = None
    step_id: str | None = None
    attempt_id: str | None = None
    tool_call_id: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "event_id",
        "session_id",
        "task_id",
        "execution_id",
        "turn_id",
        "step_id",
        "attempt_id",
        "tool_call_id",
    )
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return validate_execution_id(value, info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _utc(value, "occurred_at")

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value: Any) -> dict[str, Any]:
        checked = _strict_json(value)
        if not isinstance(checked, dict):
            raise ValueError("payload must be a strict JSON object")
        return checked

    @model_validator(mode="after")
    def validate_hierarchy(self) -> PendingSessionEvent:
        if self.step_id is not None and self.turn_id is None:
            raise ValueError("step_id requires turn_id")
        if self.attempt_id is not None and self.step_id is None:
            raise ValueError("attempt_id requires step_id")
        if self.tool_call_id is not None and self.step_id is None:
            raise ValueError("tool_call_id requires step_id")
        return self


class SessionEvent(PendingSessionEvent):
    sequence: int = Field(ge=1)


class SubmissionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    session_id: str
    message_id: str
    digest_algorithm: str
    message_digest: str
    task_id: str
    turn_id: str
    state: SubmissionState
    created_at: datetime
    updated_at: datetime

    @field_validator("session_id", "message_id", "task_id", "turn_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return validate_execution_id(value, info.field_name)

    @field_validator("digest_algorithm")
    @classmethod
    def validate_digest_algorithm(cls, value: str) -> str:
        if value != _DIGEST_ALGORITHM:
            raise ValueError(f"digest_algorithm must be {_DIGEST_ALGORITHM}")
        return value

    @field_validator("message_digest")
    @classmethod
    def validate_message_digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("message_digest must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamps(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_time_order(self) -> SubmissionRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class AgentProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str
    version: str
    system_prompt_digest: str
    model_identifier: str
    model_settings_digest: str
    skills_digest: str
    tools_digest: str
    memory_policy_version: str
    fact_schema_version: Literal[1]

    @field_validator("id", "version", "model_identifier", "memory_policy_version")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return validate_execution_id(value, info.field_name)

    @field_validator(
        "system_prompt_digest",
        "model_settings_digest",
        "skills_digest",
        "tools_digest",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be 64 lowercase hexadecimal characters")
        return value
