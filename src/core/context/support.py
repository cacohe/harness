from dataclasses import replace

from src.core.artifacts.contracts import ArtifactRef
from src.core.context.model import ModelMessage, ModelRole
from src.core.ids import ProfileId, SessionId
from src.core.persistence.journal import ConversationSummaryCheckpoint

_MAX_TOKEN_COUNT = (1 << 63) - 1
_MAX_OPTIONAL_MESSAGES = 128
_MAX_POLICY_BYTES = 256


def context_policy(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or len(value.encode("utf-8")) > _MAX_POLICY_BYTES
    ):
        raise ValueError("Context policy version is invalid")
    return value


def snapshot_summary_checkpoint(
    value: ConversationSummaryCheckpoint | None,
    *,
    session_id: SessionId | None = None,
    profile_id: ProfileId | None = None,
    policy_version: str | None = None,
) -> ConversationSummaryCheckpoint | None:
    if value is None:
        return None
    try:
        if type(value) is not ConversationSummaryCheckpoint:
            raise ValueError
        raw_session = object.__getattribute__(value, "session_id")
        raw_through = object.__getattribute__(value, "covered_through_sequence")
        raw_policy = object.__getattribute__(value, "policy_version")
        raw_profile = object.__getattribute__(value, "profile_id")
        raw_summary = object.__getattribute__(value, "summary")
        raw_from = object.__getattribute__(value, "covered_from_sequence")
        raw_artifacts = object.__getattribute__(value, "artifacts")
        if (
            type(raw_session) is not SessionId
            or type(raw_profile) is not ProfileId
            or type(raw_through) is not int
            or type(raw_from) is not int
            or type(raw_policy) is not str
            or type(raw_summary) is not str
            or type(raw_artifacts) is not tuple
            or any(type(ref) is not ArtifactRef for ref in raw_artifacts)
        ):
            raise ValueError
        result = ConversationSummaryCheckpoint(
            SessionId(object.__getattribute__(raw_session, "value")),
            raw_through,
            context_policy(raw_policy),
            ProfileId(object.__getattribute__(raw_profile, "value")),
            raw_summary,
            raw_from,
            raw_artifacts,
        )
        if (
            type(result.covered_from_sequence) is not int
            or type(result.covered_through_sequence) is not int
            or not 1
            <= result.covered_from_sequence
            <= result.covered_through_sequence
            <= _MAX_TOKEN_COUNT
            or type(result.summary) is not str
            or not result.summary.strip()
            or len(result.summary.encode("utf-8")) > 1_048_576
            or session_id is not None
            and result.session_id != session_id
            or profile_id is not None
            and result.profile_id != profile_id
            or policy_version is not None
            and result.policy_version != policy_version
        ):
            raise ValueError
        return result
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Summary checkpoint does not match the context request") from None


def summary_system_messages(
    checkpoint: ConversationSummaryCheckpoint,
) -> tuple[ModelMessage, ...]:
    messages = [ModelMessage(ModelRole.SYSTEM, checkpoint.summary)]
    if checkpoint.artifacts:
        messages.append(
            ModelMessage(
                ModelRole.SYSTEM,
                "Artifacts: " + ", ".join(ref.artifact_id for ref in checkpoint.artifacts),
            )
        )
    return tuple(messages)


def optional_system_messages(
    messages: tuple[ModelMessage, ...], name: str
) -> tuple[ModelMessage, ...]:
    if (
        type(messages) is not tuple
        or len(messages) > _MAX_OPTIONAL_MESSAGES
        or any(type(message) is not ModelMessage for message in messages)
    ):
        raise ValueError(f"{name} context requires a bounded tuple of messages")
    snapshot = tuple(replace(message) for message in messages)
    if any(message.role is not ModelRole.SYSTEM for message in snapshot):
        raise ValueError(f"{name} context must use system messages")
    return snapshot
