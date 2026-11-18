import asyncio
from datetime import datetime, timezone

import pytest
from src.adapters.persistence.memory import InMemoryRunJournal
from src.core.artifacts import ArtifactRef, ExternalizedContent, InlineContent
from src.core.commands import RunLimits, ToolCallRequest
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent, DomainEventType
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    SubmissionRecord,
    TranscriptEntry,
)
from src.core.run_receipt import validate_commit_receipt
from src.core.state import (
    ApprovalState,
    ModelCallState,
    RunState,
    RunStatus,
    StepState,
    ToolCallState,
)

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)
RUN = RunId("run")
SESSION = SessionId("session")
SUBMISSION = SubmissionId("submission")
PROFILE = ProfileId("profile")
LIMITS = RunLimits(4, 60, 256)


class EqualAll:
    def __init__(self, calls):
        self.calls = calls

    def __eq__(self, _other):
        self.calls.append("eq")
        return True

    def __hash__(self):
        self.calls.append("hash")
        return 0


def artifact():
    digest = "a" * 64
    return ArtifactRef("sha256:" + digest, "text/plain", 3, digest, metadata={"lines": 1})


def direct_batch():
    state = RunState(
        RUN,
        SESSION,
        SUBMISSION,
        PROFILE,
        LIMITS,
        2,
        status=RunStatus.COMPLETED,
        final_text="answer",
        journal_version=1,
        journal_sequence=2,
    )
    event = DomainEvent(
        EventId("event"), DomainEventType.RUN_COMPLETED, RUN, SESSION, NOW, {"ok": True}
    )
    transcript = TranscriptEntry(SESSION, RUN, SUBMISSION, "assistant", "answer")
    summary = ConversationSummaryCheckpoint(
        SESSION, 2, "v1", PROFILE, "summary", artifacts=(artifact(),)
    )
    submission = SubmissionRecord(SESSION, RUN, SUBMISSION, "digest")
    fact = ArtifactAttachmentFact(
        SESSION,
        RUN,
        SUBMISSION,
        "final",
        ArtifactAttachmentRole.FINAL_OUTPUT,
        ArtifactAttachmentAction.ATTACHED,
        artifact(),
    )
    return JournalCommit(
        state,
        events=(event,),
        transcript=(transcript,),
        summary=summary,
        submission=submission,
        artifact_facts=(fact,),
    )


def tool_batch():
    request = ToolCallRequest(
        "call",
        "write",
        '{"value":1}',
        requires_approval=True,
        approval_scope="write",
        workspace_id="workspace",
    )
    result = ExternalizedContent(InlineContent({"ok": True}, "application/json"))
    tool = ToolCallState(request, True, ApprovalState(True, resumed=True), result)
    step = StepState("step", ModelCallState("model", "", (request,)), (tool,), 1)
    state = RunState(
        RUN,
        SESSION,
        SUBMISSION,
        PROFILE,
        LIMITS,
        6,
        status=RunStatus.RUNNING,
        current_step=step,
        used_step_ids=frozenset({"step"}),
        used_tool_call_ids=frozenset({"call"}),
        journal_version=1,
        journal_sequence=0,
    )
    return JournalCommit(state)


def failed_batch():
    return JournalCommit(
        RunState(
            RUN,
            SESSION,
            SUBMISSION,
            PROFILE,
            LIMITS,
            2,
            status=RunStatus.FAILED,
            failure=RunFailure(RunErrorCode.MODEL_FAILED),
            journal_version=1,
        )
    )


def receipt(batch):
    return asyncio.run(InMemoryRunJournal().commit(batch, expected_version=0))


def mutate_id(owner, field, hook):
    identifier = object.__getattribute__(owner, field)
    object.__setattr__(identifier, "value", hook)


@pytest.mark.parametrize(
    "attack",
    [
        lambda value, hook: mutate_id(value, "session_id", hook),
        lambda value, hook: object.__setattr__(value.state, "journal_version", hook),
        lambda value, hook: mutate_id(value.state, "run_id", hook),
        lambda value, hook: mutate_id(value.state, "session_id", hook),
        lambda value, hook: mutate_id(value.state, "submission_id", hook),
        lambda value, hook: mutate_id(value.state, "profile_id", hook),
        lambda value, hook: object.__setattr__(value.events[0], "sequence", hook),
        lambda value, hook: object.__setattr__(value.events[0].event, "_payload_json", hook),
        lambda value, hook: object.__setattr__(value.events[0].event, "timestamp", hook),
        lambda value, hook: mutate_id(value.events[0].event, "event_id", hook),
        lambda value, hook: mutate_id(value.events[0].event, "run_id", hook),
        lambda value, hook: mutate_id(value.events[0].event, "session_id", hook),
        lambda value, hook: object.__setattr__(value.transcript[0], "content", hook),
        lambda value, hook: mutate_id(value.transcript[0], "session_id", hook),
        lambda value, hook: mutate_id(value.transcript[0], "run_id", hook),
        lambda value, hook: mutate_id(value.transcript[0], "submission_id", hook),
        lambda value, hook: mutate_id(value.summaries[0], "session_id", hook),
        lambda value, hook: mutate_id(value.summaries[0], "profile_id", hook),
        lambda value, hook: object.__setattr__(
            value.summaries[0].artifacts[0], "artifact_id", hook
        ),
        lambda value, hook: object.__setattr__(value.submission, "digest", hook),
        lambda value, hook: mutate_id(value.submission, "session_id", hook),
        lambda value, hook: mutate_id(value.submission, "run_id", hook),
        lambda value, hook: mutate_id(value.submission, "submission_id", hook),
    ],
)
def test_receipt_rebuild_rejects_nested_equality_hooks_without_executing_them(attack):
    batch = direct_batch()
    forged = receipt(batch)
    calls = []
    attack(forged, EqualAll(calls))

    with pytest.raises(JournalError) as error:
        validate_commit_receipt(forged, batch, expected_version=0, previous_sequence=0)

    assert error.value.failure == RunFailure(RunErrorCode.INVARIANT_VIOLATION)
    assert calls == []


@pytest.mark.parametrize(
    "field",
    [
        "tool_call_id",
        "name",
        "arguments_json",
        "requires_approval",
        "approval_scope",
        "workspace_id",
    ],
)
def test_receipt_rebuild_rejects_tampered_tool_request_fields_without_equality(field):
    batch = tool_batch()
    forged = receipt(batch)
    calls = []
    request = forged.state.current_step.model_call.tool_requests[0]
    object.__setattr__(request, field, EqualAll(calls))

    with pytest.raises(JournalError):
        validate_commit_receipt(forged, batch, expected_version=0, previous_sequence=0)

    assert calls == []


@pytest.mark.parametrize(
    "attack",
    [
        lambda state, hook: object.__setattr__(state.limits, "max_steps", hook),
        lambda state, hook: object.__setattr__(state.current_step, "step_id", hook),
        lambda state, hook: object.__setattr__(state.current_step, "next_tool_index", hook),
        lambda state, hook: object.__setattr__(
            state.current_step.model_call, "model_call_id", hook
        ),
        lambda state, hook: object.__setattr__(state.current_step.model_call, "result", hook),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0], "requires_approval", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].approval, "approved", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].approval, "resumed", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].result, "summary", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].result._inline_content, "media_type", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].result._inline_content, "size_bytes", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].result._inline_content, "_encoded", hook
        ),
        lambda state, hook: object.__setattr__(
            state.current_step.tool_calls[0].result._inline_content, "_is_binary", hook
        ),
    ],
)
def test_receipt_rebuild_rejects_all_nested_run_state_hooks(attack):
    batch = tool_batch()
    forged = receipt(batch)
    calls = []
    attack(forged.state, EqualAll(calls))

    with pytest.raises(JournalError):
        validate_commit_receipt(forged, batch, expected_version=0, previous_sequence=0)

    assert calls == []


def test_receipt_rebuild_rejects_missing_nested_fields_without_descriptor_access():
    batch = tool_batch()
    forged = receipt(batch)
    object.__delattr__(forged.state.current_step.model_call.tool_requests[0], "name")
    with pytest.raises(JournalError):
        validate_commit_receipt(forged, batch, expected_version=0, previous_sequence=0)


def test_receipt_rebuild_rejects_tampered_run_failure_message_without_equality():
    batch = failed_batch()
    forged = receipt(batch)
    calls = []
    object.__setattr__(forged.state.failure, "message", EqualAll(calls))
    with pytest.raises(JournalError):
        validate_commit_receipt(forged, batch, expected_version=0, previous_sequence=0)
    assert calls == []


def test_receipt_rebuild_returns_an_independent_trusted_snapshot():
    batch = direct_batch()
    original = receipt(batch)
    rebuilt = validate_commit_receipt(original, batch, expected_version=0, previous_sequence=0)
    assert rebuilt == original
    assert rebuilt is not original
    assert rebuilt.state is not original.state
    assert rebuilt.events[0] is not original.events[0]
    assert rebuilt.transcript[0] is not original.transcript[0]
    assert rebuilt.summaries[0] is not original.summaries[0]
    assert rebuilt.submission is not original.submission
    assert rebuilt.artifact_facts == original.artifact_facts
    assert rebuilt.artifact_facts[0] is not original.artifact_facts[0]


def test_receipt_rejects_tampered_artifact_attachment_descriptor() -> None:
    batch = direct_batch()
    forged = receipt(batch)
    object.__setattr__(forged.artifact_facts[0].artifact, "size_bytes", 99)
    with pytest.raises(JournalError):
        validate_commit_receipt(forged, batch, expected_version=0, previous_sequence=0)
