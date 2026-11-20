import json
from datetime import datetime, timezone
from typing import TypeVar

from src.core.artifacts import (
    ArtifactContent,
    ArtifactRef,
    ExternalizedContent,
    InlineContent,
    _decode_json,
    _decode_metadata,
    _is_json_media_type,
)
from src.core.commands import RunLimits, ToolCallRequest, snapshot_tool_call
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId, _Id
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ConversationSummaryCheckpoint,
    JournalEvent,
    JournalSnapshot,
    SubmissionRecord,
    TranscriptEntry,
    TranscriptEntryType,
)
from src.core.state import (
    ApprovalState,
    ModelCallState,
    RunState,
    RunStatus,
    StepState,
    ToolCallState,
)

_IdValue = TypeVar("_IdValue", bound=_Id)


def snapshot_journal_snapshot(value: object) -> JournalSnapshot:
    _require_type(value, JournalSnapshot)
    state_value = _raw(value, "state")
    if type(state_value) is not RunState:
        raise ValueError
    events = _raw_tuple(value, "events", JournalEvent, _snapshot_journal_event)
    transcript = _raw_tuple(value, "transcript", TranscriptEntry, snapshot_transcript)
    summaries = _raw_tuple(value, "summaries", ConversationSummaryCheckpoint, snapshot_summary)
    submission_value = _raw(value, "submission")
    submission = None
    if submission_value is not None:
        submission = snapshot_submission(submission_value)
    version = _exact_int(_raw(value, "version"))
    sequence = _exact_int(_raw(value, "sequence"))
    if version < 0 or sequence < 0:
        raise ValueError
    artifact_facts = _raw_tuple(
        value, "artifact_facts", ArtifactAttachmentFact, snapshot_artifact_attachment
    )
    return JournalSnapshot(
        _snapshot_id(_raw(value, "session_id"), SessionId),
        version,
        sequence,
        snapshot_run_state(state_value),
        events,
        transcript,
        summaries,
        submission,
        artifact_facts,
    )


def snapshot_artifact_attachment(value: object) -> ArtifactAttachmentFact:
    _require_type(value, ArtifactAttachmentFact)
    role = _raw(value, "role")
    action = _raw(value, "action")
    if type(role) is not ArtifactAttachmentRole or type(action) is not ArtifactAttachmentAction:
        raise ValueError
    return ArtifactAttachmentFact(
        _snapshot_id(_raw(value, "session_id"), SessionId),
        _snapshot_id(_raw(value, "run_id"), RunId),
        _snapshot_id(_raw(value, "submission_id"), SubmissionId),
        _exact_text(_raw(value, "owner_id")),
        role,
        action,
        _snapshot_artifact(_raw(value, "artifact")),
    )


def snapshot_run_state(value: object) -> RunState:
    _require_type(value, RunState)
    step_value = _raw(value, "current_step")
    step = None if step_value is None else _snapshot_step(step_value)
    final_artifacts = _raw_tuple(value, "final_artifacts", ArtifactRef, _snapshot_artifact)
    failure_value = _raw(value, "failure")
    failure = None if failure_value is None else _snapshot_failure(failure_value)
    used_steps = _snapshot_string_set(_raw(value, "used_step_ids"))
    used_tools = _snapshot_string_set(_raw(value, "used_tool_call_ids"))
    status = _snapshot_run_status(_raw(value, "status"))
    elapsed = _raw(value, "elapsed_seconds")
    if type(elapsed) is not int and type(elapsed) is not float:
        raise ValueError
    final_text = _raw(value, "final_text")
    if final_text is not None and type(final_text) is not str:
        raise ValueError
    return RunState(
        run_id=_snapshot_id(_raw(value, "run_id"), RunId),
        session_id=_snapshot_id(_raw(value, "session_id"), SessionId),
        submission_id=_snapshot_id(_raw(value, "submission_id"), SubmissionId),
        profile_id=_snapshot_id(_raw(value, "profile_id"), ProfileId),
        limits=_snapshot_limits(_raw(value, "limits")),
        version=_exact_int(_raw(value, "version")),
        status=status,
        current_step=step,
        elapsed_seconds=float(elapsed),
        used_step_ids=used_steps,
        used_tool_call_ids=used_tools,
        final_text=final_text,
        final_artifacts=final_artifacts,
        failure=failure,
        journal_version=_exact_int(_raw(value, "journal_version")),
        journal_sequence=_exact_int(_raw(value, "journal_sequence")),
    )


def snapshot_transcript(value: object, *, sequence: int | None = None) -> TranscriptEntry:
    _require_type(value, TranscriptEntry)
    role = _exact_text(_raw(value, "role"))
    content = _exact_text(_raw(value, "content"), allow_empty=True)
    raw_sequence = _exact_int(_raw(value, "sequence"))
    entry_type = _raw(value, "entry_type")
    format_version = _exact_int(_raw(value, "format_version"))
    if (
        raw_sequence < 0
        or not role
        or type(entry_type) is not TranscriptEntryType
        or format_version != 1
    ):
        raise ValueError
    return TranscriptEntry(
        _snapshot_id(_raw(value, "session_id"), SessionId),
        _snapshot_id(_raw(value, "run_id"), RunId),
        _snapshot_id(_raw(value, "submission_id"), SubmissionId),
        role,
        content,
        raw_sequence if sequence is None else sequence,
        entry_type,
        format_version,
    )


def snapshot_summary(value: object) -> ConversationSummaryCheckpoint:
    _require_type(value, ConversationSummaryCheckpoint)
    return ConversationSummaryCheckpoint(
        _snapshot_id(_raw(value, "session_id"), SessionId),
        _exact_int(_raw(value, "covered_through_sequence")),
        _exact_text(_raw(value, "policy_version")),
        _snapshot_id(_raw(value, "profile_id"), ProfileId),
        _exact_text(_raw(value, "summary")),
        covered_from_sequence=_exact_int(_raw(value, "covered_from_sequence")),
        artifacts=_raw_tuple(value, "artifacts", ArtifactRef, _snapshot_artifact),
    )


def snapshot_submission(value: object) -> SubmissionRecord:
    _require_type(value, SubmissionRecord)
    digest = _exact_text(_raw(value, "digest"))
    if not digest:
        raise ValueError
    return SubmissionRecord(
        _snapshot_id(_raw(value, "session_id"), SessionId),
        _snapshot_id(_raw(value, "run_id"), RunId),
        _snapshot_id(_raw(value, "submission_id"), SubmissionId),
        digest,
    )


def snapshot_domain_event(value: object) -> DomainEvent:
    _require_type(value, DomainEvent)
    event_type = _snapshot_event_type(_raw(value, "type"))
    timestamp = _snapshot_timestamp(_raw(value, "timestamp"))
    payload_json = _exact_text(_raw(value, "_payload_json"), allow_empty=True)
    payload = json.loads(payload_json)
    if type(payload) is not dict:
        raise ValueError
    rebuilt = DomainEvent(
        _snapshot_id(_raw(value, "event_id"), EventId),
        event_type,
        _snapshot_id(_raw(value, "run_id"), RunId),
        _snapshot_id(_raw(value, "session_id"), SessionId),
        timestamp,
        payload,
    )
    if object.__getattribute__(rebuilt, "_payload_json") != payload_json:
        raise ValueError
    return rebuilt


def _snapshot_journal_event(value: object) -> JournalEvent:
    _require_type(value, JournalEvent)
    sequence = _exact_int(_raw(value, "sequence"))
    if sequence < 1:
        raise ValueError
    return JournalEvent(sequence, snapshot_domain_event(_raw(value, "event")))


def _snapshot_step(value: object) -> StepState:
    _require_type(value, StepState)
    model_value = _raw(value, "model_call")
    model = None if model_value is None else _snapshot_model_call(model_value)
    return StepState(
        _exact_text(_raw(value, "step_id")),
        model,
        _raw_tuple(value, "tool_calls", ToolCallState, _snapshot_tool_state),
        _exact_int(_raw(value, "next_tool_index")),
    )


def _snapshot_model_call(value: object) -> ModelCallState:
    _require_type(value, ModelCallState)
    result = _raw(value, "result")
    if result is not None and type(result) is not str:
        raise ValueError
    return ModelCallState(
        _exact_text(_raw(value, "model_call_id")),
        result,
        _raw_tuple(value, "tool_requests", ToolCallRequest, snapshot_tool_call),
    )


def _snapshot_tool_state(value: object) -> ToolCallState:
    _require_type(value, ToolCallState)
    approval_value = _raw(value, "approval")
    approval = None if approval_value is None else _snapshot_approval(approval_value)
    result_value = _raw(value, "result")
    result: ExternalizedContent | RunFailure | None = None
    if type(result_value) is ExternalizedContent:
        result = _snapshot_externalized(result_value)
    elif type(result_value) is RunFailure:
        result = _snapshot_failure(result_value)
    elif result_value is not None:
        raise ValueError
    required = _raw(value, "requires_approval")
    if type(required) is not bool:
        raise ValueError
    return ToolCallState(
        snapshot_tool_call(_raw(value, "request")),
        required,
        approval,
        result,
    )


def _snapshot_approval(value: object) -> ApprovalState:
    _require_type(value, ApprovalState)
    approved = _raw(value, "approved")
    resumed = _raw(value, "resumed")
    if (approved is not None and type(approved) is not bool) or type(resumed) is not bool:
        raise ValueError
    return ApprovalState(approved, resumed)


def _snapshot_externalized(value: object) -> ExternalizedContent:
    _require_type(value, ExternalizedContent)
    inline_value = _raw(value, "_inline_content")
    inline = None if inline_value is None else _snapshot_inline(inline_value)
    artifacts = _raw_tuple(value, "_artifacts", ArtifactRef, _snapshot_artifact)
    summary = _raw(value, "summary")
    if summary is not None and type(summary) is not str:
        raise ValueError
    return ExternalizedContent(inline, artifacts, summary)


def _snapshot_inline(value: object) -> InlineContent:
    _require_type(value, InlineContent)
    media_type = _exact_text(_raw(value, "media_type"))
    size = _exact_int(_raw(value, "size_bytes"))
    encoded = _raw(value, "_encoded")
    is_binary = _raw(value, "_is_binary")
    if type(encoded) is not bytes or type(is_binary) is not bool or len(encoded) != size:
        raise ValueError
    content: ArtifactContent
    if is_binary:
        content = encoded
    elif _is_json_media_type(media_type):
        content = _decode_json(encoded)
    else:
        content = encoded.decode("utf-8")
    rebuilt = InlineContent(content, media_type)
    if object.__getattribute__(rebuilt, "_encoded") != encoded:
        raise ValueError
    return rebuilt


def _snapshot_artifact(value: object) -> ArtifactRef:
    _require_type(value, ArtifactRef)
    metadata_json = _exact_text(_raw(value, "_metadata_json"))
    rebuilt = ArtifactRef(
        _exact_text(_raw(value, "artifact_id")),
        _exact_text(_raw(value, "media_type")),
        _exact_int(_raw(value, "size_bytes")),
        _exact_text(_raw(value, "sha256")),
        _optional_text(_raw(value, "name")),
        _decode_metadata(metadata_json),
    )
    if object.__getattribute__(rebuilt, "_metadata_json") != metadata_json:
        raise ValueError
    return rebuilt


def _snapshot_failure(value: object) -> RunFailure:
    _require_type(value, RunFailure)
    message = _exact_text(_raw(value, "message"))
    result = RunFailure(_snapshot_error_code(_raw(value, "code")))
    if result.message != message:
        raise ValueError
    return result


def _snapshot_limits(value: object) -> RunLimits:
    _require_type(value, RunLimits)
    elapsed = _raw(value, "max_elapsed_seconds")
    if type(elapsed) is not int and type(elapsed) is not float:
        raise ValueError
    return RunLimits(
        _exact_int(_raw(value, "max_steps")),
        float(elapsed),
        _exact_int(_raw(value, "max_inline_result_bytes")),
    )


def _snapshot_id(value: object, expected: type[_IdValue]) -> _IdValue:
    _require_type(value, expected)
    return expected(_exact_text(_raw(value, "value")))


def _snapshot_run_status(value: object) -> RunStatus:
    _require_type(value, RunStatus)
    result = RunStatus(_exact_text(_raw(value, "_value_")))
    if result is not value:
        raise ValueError
    return result


def _snapshot_event_type(value: object) -> DomainEventType:
    _require_type(value, DomainEventType)
    result = DomainEventType(_exact_text(_raw(value, "_value_")))
    if result is not value:
        raise ValueError
    return result


def _snapshot_error_code(value: object) -> RunErrorCode:
    _require_type(value, RunErrorCode)
    result = RunErrorCode(_exact_text(_raw(value, "_value_")))
    if result is not value:
        raise ValueError
    return result


def _snapshot_timestamp(value: object) -> datetime:
    if (
        type(value) is not datetime
        or object.__getattribute__(value, "tzinfo") is not timezone.utc
        or object.__getattribute__(value, "fold") != 0
    ):
        raise ValueError
    return datetime(
        object.__getattribute__(value, "year"),
        object.__getattribute__(value, "month"),
        object.__getattribute__(value, "day"),
        object.__getattribute__(value, "hour"),
        object.__getattribute__(value, "minute"),
        object.__getattribute__(value, "second"),
        object.__getattribute__(value, "microsecond"),
        timezone.utc,
        fold=0,
    )


def _snapshot_string_set(value: object) -> frozenset[str]:
    if type(value) is not frozenset or any(type(item) is not str for item in value):
        raise ValueError
    return frozenset(value)


def _raw_tuple(value: object, name: str, expected: type, snapshot):
    items = _raw(value, name)
    if type(items) is not tuple or any(type(item) is not expected for item in items):
        raise ValueError
    return tuple(snapshot(item) for item in items)


def _raw(value: object, name: str):
    return object.__getattribute__(value, name)


def _require_type(value: object, expected: type) -> None:
    if type(value) is not expected:
        raise ValueError


def _exact_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError
    return value


def _exact_text(value: object, *, allow_empty: bool = False) -> str:
    if type(value) is not str or not allow_empty and not value:
        raise ValueError
    value.encode("utf-8")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _exact_text(value)
