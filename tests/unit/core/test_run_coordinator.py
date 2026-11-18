import asyncio
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.core.artifacts import InlineContent
from src.core.commands import (
    AdvanceRun,
    CancelRun,
    CommandContext,
    DecideToolCall,
    ResumeRun,
    RunLimits,
    SubmitRun,
)
from src.core.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextRequest,
    ContextWindow,
    EstimateQuality,
    TokenEstimate,
)
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import EventId, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import ConversationSummaryCheckpoint, JournalError
from src.core.model import ModelMessage, ModelResponse, ModelRole, ModelToolCall
from src.core.run import RunCoordinator
from src.core.state import RunState, RunStatus
from src.core.tools import ToolApproval, ToolResult, ToolResultStatus, ToolSpec

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)
RUN = RunId("run")
SESSION = SessionId("session")
SUBMISSION = SubmissionId("submission")
PROFILE = ProfileId("profile")
LIMITS = RunLimits(8, 60, 64)
BUDGET = ContextBudget(100, 10, 5)


class Clock:
    def __init__(self):
        self.tick = 0.0

    def now(self):
        return NOW

    def monotonic(self):
        return self.tick


class Ids:
    def __init__(self):
        self.count = 0

    def event(self):
        self.count += 1
        return EventId(f"event-{self.count}")

    def step(self):
        self.count += 1
        return f"step-{self.count}"

    def model_call(self):
        self.count += 1
        return f"model-{self.count}"


class Context:
    def __init__(self, calls, failure=None, checkpoint=None):
        self.calls = calls
        self.failure = failure
        self.checkpoint = checkpoint

    async def build(self, request, budget):
        self.calls.append("context")
        if self.failure:
            raise self.failure
        return ContextWindow(
            (request.system_message, request.current_input),
            request.tools,
            TokenEstimate(2, EstimateQuality.EXACT),
            budget,
            (),
            request.policy_version,
            self.checkpoint,
        )


class Model:
    def __init__(self, calls, responses):
        self.calls = calls
        self.responses = deque(responses)

    async def complete(self, request):
        self.calls.append("model")
        value = self.responses.popleft()
        if isinstance(value, BaseException):
            raise value
        return value


class Tools:
    def __init__(self, calls, results):
        self.calls = calls
        self.results = deque(results)
        self.requests = []

    async def execute(self, request):
        self.calls.append(f"tool:{request.tool_call_id}")
        self.requests.append(request)
        value = self.results.popleft()
        if isinstance(value, BaseException):
            raise value
        if type(value) is ToolResult and value.execution_key is None:
            return ToolResult(
                value.status,
                inline_content=value.inline_content,
                artifacts=value.artifacts,
                summary=value.summary,
                failure=value.failure,
                execution_key=request.idempotency_key,
            )
        return value


class Journal(InMemoryRunJournal):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls
        self.batches = []

    async def commit(self, batch, expected_version):
        self.calls.append("journal")
        self.batches.append((batch, expected_version))
        return await super().commit(batch, expected_version)


def command_context(name="external"):
    return CommandContext(NOW, 0, (EventId(name),))


def submit():
    return SubmitRun(
        RUN,
        SESSION,
        SUBMISSION,
        PROFILE,
        "question",
        context=command_context(),
    )


def request(tools=()):
    return ContextRequest(
        SESSION,
        PROFILE,
        "v1",
        ModelMessage(ModelRole.SYSTEM, "system"),
        ModelMessage(ModelRole.USER, "question"),
        tools,
    )


def initial():
    return RunState.create(submit(), LIMITS)


def coordinator(
    *, responses, specs=(), tool_results=(), context_failure=None, checkpoint=None, journal=None
):
    calls = []
    journal = journal or Journal(calls)
    repository = InMemoryArtifactRepository(1024)
    externalizer = PolicyArtifactExternalizer(
        repository,
        ArtifactPolicy(8, 1024, 64, frozenset({"application/json", "text/plain"})),
    )
    tools = Tools(calls, tool_results)
    result = RunCoordinator(
        context=Context(calls, context_failure, checkpoint),
        model=Model(calls, responses),
        tools=tools,
        artifacts=externalizer,
        journal=journal,
        clock=Clock(),
        ids=Ids(),
        context_request=request(specs),
        context_budget=BUDGET,
    )
    return result, calls, journal, tools, repository


def run(awaitable):
    return asyncio.run(awaitable)


def test_direct_answer_orders_ports_and_commits_terminal_state_once():
    coordinator_, calls, journal, _, _ = coordinator(responses=[ModelResponse("answer")])

    outcome = run(coordinator_.advance(initial(), submit()))

    assert calls == ["context", "model", "journal"]
    assert outcome.state.status is RunStatus.COMPLETED
    assert outcome.state.final_text == "answer"
    assert outcome.journal_version == 1
    assert len(journal.batches) == 1
    assert [entry.role for entry in journal.batches[0][0].transcript] == ["assistant"]


def test_large_direct_answer_becomes_final_artifact_reference_and_bounded_summary():
    answer = "x" * 80
    coordinator_, _, journal, _, repository = coordinator(responses=[ModelResponse(answer)])
    outcome = run(coordinator_.advance(initial(), submit()))
    assert len(outcome.state.final_text.encode("utf-8")) <= LIMITS.max_inline_result_bytes
    assert len(outcome.state.final_artifacts) == 1
    ref = outcome.state.final_artifacts[0]
    assert run(repository.stat(ref.artifact_id)) == ref
    assert answer not in journal.batches[0][0].transcript[0].content


def test_context_budget_failure_does_not_call_model_and_is_committed():
    coordinator_, calls, _, _, _ = coordinator(
        responses=[], context_failure=ContextBudgetExceeded()
    )

    outcome = run(coordinator_.advance(initial(), submit()))

    assert calls == ["context", "journal"]
    assert outcome.state.failure == RunFailure(RunErrorCode.CONTEXT_BUDGET_EXCEEDED)


def test_elapsed_limit_is_committed_before_any_adapter_call():
    coordinator_, calls, _, _, _ = coordinator(responses=[])
    over_limit = replace(submit(), context=CommandContext(NOW, 61, (EventId("limit"),)))
    outcome = run(coordinator_.advance(initial(), over_limit))
    assert outcome.state.failure == RunFailure(RunErrorCode.LIMIT_EXCEEDED)
    assert calls == ["journal"]


def test_model_failure_hides_adapter_exception_text():
    coordinator_, _, _, _, _ = coordinator(responses=[RuntimeError(r"C:\secret\token")])
    outcome = run(coordinator_.advance(initial(), submit()))
    assert outcome.state.failure == RunFailure(RunErrorCode.MODEL_FAILED)
    assert "secret" not in outcome.state.failure.message


@pytest.mark.parametrize("bad_result", [object(), "not-a-response"])
def test_untrusted_model_gateway_result_is_stably_classified(bad_result):
    coordinator_, _, _, _, _ = coordinator(responses=[bad_result])
    outcome = run(coordinator_.advance(initial(), submit()))
    assert outcome.state.failure == RunFailure(RunErrorCode.MODEL_FAILED)


def test_tampered_model_response_fields_do_not_leak_runtime_errors():
    response = ModelResponse("answer")
    object.__setattr__(response, "text", object())
    coordinator_, _, _, _, _ = coordinator(responses=[response])
    outcome = run(coordinator_.advance(initial(), submit()))
    assert outcome.state.failure == RunFailure(RunErrorCode.MODEL_FAILED)


def test_model_gateway_descriptor_and_tuple_alias_are_never_trusted():
    class DescriptorResponse:
        __dataclass_fields__ = {"text": object()}

        @property
        def text(self):
            raise RuntimeError(r"C:\secret\descriptor")

    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    object.__setattr__(response, "tool_calls", [response.tool_calls[0]])
    for untrusted in (DescriptorResponse(), response):
        coordinator_, _, _, _, _ = coordinator(responses=[untrusted])
        outcome = run(coordinator_.advance(initial(), submit()))
        assert outcome.state.failure == RunFailure(RunErrorCode.MODEL_FAILED)


def test_ordered_tool_batch_pauses_and_resumes_without_replaying_side_effects():
    specs = (
        ToolSpec("first", "first", "{}"),
        ToolSpec(
            "write",
            "write",
            "{}",
            requires_approval=True,
            approval_scope="write",
            workspace_id="workspace",
        ),
        ToolSpec("last", "last", "{}"),
    )
    response = ModelResponse(
        tool_calls=tuple(ModelToolCall(name, name, "{}") for name in ("first", "write", "last"))
    )
    success = ToolResult(
        ToolResultStatus.SUCCESS,
        inline_content=InlineContent({"ok": True}, "application/json"),
    )
    coordinator_, calls, journal, tools, _ = coordinator(
        responses=[response], specs=specs, tool_results=[success, success]
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    waiting = run(
        coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools")))
    ).state
    assert waiting.status is RunStatus.WAITING_APPROVAL
    assert calls == ["context", "model", "journal", "tool:first", "journal"]
    assert waiting.current_step is not None
    assert tuple(tool.request.tool_call_id for tool in waiting.current_step.tool_calls) == (
        "first",
        "write",
    )
    assert waiting.current_step.next_tool_index == 1
    restored = run(journal.load_head(SESSION)).state
    assert restored == waiting

    decided = run(
        coordinator_.advance(
            restored,
            DecideToolCall(RUN, SESSION, "write", True, context=command_context("decision")),
        )
    ).state
    resumed_coordinator, _, _, resumed_tools, _ = coordinator(
        responses=[],
        specs=specs,
        tool_results=[success, success],
        journal=journal,
    )
    resumed = run(
        resumed_coordinator.advance(
            decided, ResumeRun(RUN, SESSION, context=command_context("resume"))
        )
    ).state

    assert resumed.status is RunStatus.RUNNING
    assert resumed.current_step is None
    requests = [*tools.requests, *resumed_tools.requests]
    assert [item.tool_call_id for item in requests] == ["first", "write", "last"]
    assert [item.approval for item in requests] == [
        ToolApproval.NOT_REQUIRED,
        ToolApproval.APPROVED,
        ToolApproval.NOT_REQUIRED,
    ]
    assert (requests[1].approval_scope, requests[1].workspace_id) == (
        "write",
        "workspace",
    )
    assert len(journal.batches) == 4


def test_model_tool_batch_is_committed_before_any_tool_gateway_call():
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    success = ToolResult(
        ToolResultStatus.SUCCESS,
        inline_content=InlineContent({"ok": True}, "application/json"),
    )
    coordinator_, calls, journal, tools, _ = coordinator(
        responses=[response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[success],
    )

    model_outcome = run(coordinator_.advance(initial(), submit()))

    assert calls == ["context", "model", "journal"]
    assert tools.requests == []
    assert model_outcome.state.current_step is not None
    assert model_outcome.state.current_step.model_call is not None
    assert [
        item.tool_call_id for item in model_outcome.state.current_step.model_call.tool_requests
    ] == ["call"]
    assert journal.batches[0][0].state == model_outcome.state


def test_model_batch_commit_failure_never_calls_tool_and_retry_does_not_recall_model():
    class FailOnceJournal(Journal):
        failed = False

        async def commit(self, batch, expected_version):
            if not self.failed:
                self.failed = True
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            return await super().commit(batch, expected_version)

    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    journal = FailOnceJournal([])
    coordinator_, calls, _, tools, _ = coordinator(
        responses=[response, response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[],
        journal=journal,
    )
    with pytest.raises(JournalError):
        run(coordinator_.advance(initial(), submit()))
    assert tools.requests == []
    assert calls == ["context", "model"]
    durable = run(coordinator_.advance(initial(), submit())).state
    assert durable.current_step is not None
    assert calls.count("model") == 2
    # Once the model batch is durable, tool retry uses it and never calls model again.
    calls.clear()
    coordinator_._tools.results.append(
        ToolResult(
            ToolResultStatus.SUCCESS,
            inline_content=InlineContent({"ok": True}, "application/json"),
        )
    )
    run(coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools"))))
    assert "model" not in calls


def test_persisted_approval_requirement_cannot_be_downgraded_by_new_catalog():
    strict = ToolSpec(
        "write",
        "write",
        "{}",
        requires_approval=True,
        approval_scope="write",
        workspace_id="workspace",
    )
    response = ModelResponse(tool_calls=(ModelToolCall("call", "write", "{}"),))
    first, _, journal, _, _ = coordinator(responses=[response], specs=(strict,))
    durable = run(first.advance(initial(), submit())).state
    relaxed = replace(strict, requires_approval=False)
    resumed, calls, _, tools, _ = coordinator(responses=[], specs=(relaxed,), journal=journal)

    waiting = run(
        resumed.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools")))
    ).state

    assert waiting.status is RunStatus.WAITING_APPROVAL
    assert tools.requests == []
    assert calls == []


def test_denied_tool_settles_and_later_calls_continue_without_gateway_dispatch():
    specs = (
        ToolSpec(
            "write",
            "write",
            "{}",
            requires_approval=True,
            approval_scope="write",
            workspace_id="workspace",
        ),
        ToolSpec("read", "read", "{}"),
    )
    response = ModelResponse(
        tool_calls=(ModelToolCall("write", "write", "{}"), ModelToolCall("read", "read", "{}"))
    )
    success = ToolResult(
        ToolResultStatus.SUCCESS, inline_content=InlineContent({"value": 1}, "application/json")
    )
    coordinator_, _, _, tools, _ = coordinator(
        responses=[response], specs=specs, tool_results=[success]
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    waiting = run(
        coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools")))
    ).state
    denied = run(
        coordinator_.advance(
            waiting,
            DecideToolCall(RUN, SESSION, "write", False, context=command_context("deny")),
        )
    ).state
    settled = run(
        coordinator_.advance(denied, ResumeRun(RUN, SESSION, context=command_context("resume")))
    ).state
    assert settled.current_step is None
    assert [item.tool_call_id for item in tools.requests] == ["read"]


def test_large_tool_result_is_externalized_before_journal_reference():
    spec = ToolSpec("read_artifact", "read", "{}")
    raw = ToolResult(
        ToolResultStatus.SUCCESS,
        inline_content=InlineContent({"payload": "x" * 80}, "application/json"),
    )
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read_artifact", "{}"),))
    coordinator_, calls, journal, _, repository = coordinator(
        responses=[response], specs=(spec,), tool_results=[raw]
    )

    durable = run(coordinator_.advance(initial(), submit())).state
    outcome = run(
        coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools")))
    )

    assert outcome.state.current_step is None
    tool_entry = journal.batches[-1][0].transcript[-1]
    assert "sha256:" in tool_entry.content
    assert "x" * 80 not in tool_entry.content
    assert calls[-1] == "journal"
    artifact_id = "sha256:" + tool_entry.content.split("sha256:", 1)[1][:64]
    assert run(repository.stat(artifact_id)) is not None


def test_terminal_repeat_returns_without_any_port_io():
    coordinator_, calls, _, _, _ = coordinator(responses=[ModelResponse("answer")])
    terminal = run(coordinator_.advance(initial(), submit())).state
    calls.clear()
    outcome = run(
        coordinator_.advance(terminal, AdvanceRun(RUN, SESSION, context=command_context("again")))
    )
    assert outcome.state == terminal
    assert calls == []


def test_native_model_cancellation_propagates_without_commit():
    coordinator_, calls, journal, _, _ = coordinator(responses=[asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        run(coordinator_.advance(initial(), submit()))
    assert calls == ["context", "model"]
    assert journal.batches == []


def test_native_tool_cancellation_propagates_without_committing_side_effect_state():
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    coordinator_, calls, journal, _, _ = coordinator(
        responses=[response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[asyncio.CancelledError()],
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    with pytest.raises(asyncio.CancelledError):
        run(
            coordinator_.advance(
                durable, AdvanceRun(RUN, SESSION, context=command_context("tools"))
            )
        )
    assert calls == ["context", "model", "journal", "tool:call"]
    assert len(journal.batches) == 1


@pytest.mark.parametrize("bad_result", [object(), "not-a-result"])
def test_untrusted_tool_gateway_result_is_stably_classified(bad_result):
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    coordinator_, _, journal, _, _ = coordinator(
        responses=[response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[bad_result],
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    outcome = run(
        coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools")))
    )
    assert outcome.state.current_step is None
    assert outcome.state.failure is None
    assert "tool_failed" in journal.batches[-1][0].transcript[-1].content


def test_tool_result_identity_mismatch_is_stably_rejected():
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    result = ToolResult(
        ToolResultStatus.SUCCESS,
        inline_content=InlineContent({"unsafe": True}, "application/json"),
        execution_key="0" * 64,
    )
    coordinator_, _, journal, _, _ = coordinator(
        responses=[response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[result],
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    run(coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools"))))
    assert "tool_failed" in journal.batches[-1][0].transcript[-1].content


def test_tampered_tool_result_fields_do_not_leak_runtime_errors():
    result = ToolResult(
        ToolResultStatus.SUCCESS,
        inline_content=InlineContent({"unsafe": True}, "application/json"),
        execution_key="a" * 64,
    )
    object.__setattr__(result, "_content", object())
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    coordinator_, _, journal, _, _ = coordinator(
        responses=[response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[result],
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    run(coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools"))))
    assert "tool_failed" in journal.batches[-1][0].transcript[-1].content


def test_tool_commit_retry_uses_stable_key_without_repeating_side_effect():
    class FailToolCommitOnce(Journal):
        attempts = 0

        async def commit(self, batch, expected_version):
            self.attempts += 1
            if self.attempts == 2:
                self.calls.append("journal")
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            return await super().commit(batch, expected_version)

    class IdempotentTools:
        def __init__(self, calls):
            self.calls = calls
            self.requests = []
            self.cache = {}
            self.side_effects = 0

        async def execute(self, request):
            self.calls.append(f"tool:{request.tool_call_id}")
            self.requests.append(request)
            cached = self.cache.get(request.idempotency_key)
            if cached is not None:
                return cached
            self.side_effects += 1
            result = ToolResult(
                ToolResultStatus.SUCCESS,
                inline_content=InlineContent({"effect": self.side_effects}, "application/json"),
                execution_key=request.idempotency_key,
            )
            self.cache[request.idempotency_key] = result
            return result

    journal = FailToolCommitOnce([])
    coordinator_, calls, _, _, _ = coordinator(
        responses=[ModelResponse(tool_calls=(ModelToolCall("call", "write", "{}"),))],
        specs=(ToolSpec("write", "write", "{}"),),
        journal=journal,
    )
    tools = IdempotentTools(calls)
    coordinator_._tools = tools
    durable = run(coordinator_.advance(initial(), submit())).state
    command = AdvanceRun(RUN, SESSION, context=command_context("tools"))

    with pytest.raises(JournalError):
        run(coordinator_.advance(durable, command))
    outcome = run(coordinator_.advance(durable, command))

    assert outcome.state.current_step is None
    assert tools.side_effects == 1
    assert len(tools.requests) == 2
    assert tools.requests[0].idempotency_key == tools.requests[1].idempotency_key
    assert '"effect":1' in journal.batches[-1][0].transcript[-1].content


def test_explicit_cancel_skips_context_model_and_tool_ports():
    running = replace(initial(), status=RunStatus.RUNNING, version=1)
    coordinator_, calls, _, _, _ = coordinator(responses=[])
    outcome = run(
        coordinator_.advance(running, CancelRun(RUN, SESSION, context=command_context("cancel")))
    )
    assert outcome.state.status is RunStatus.CANCELLED
    assert calls == ["journal"]


def test_journal_conflict_is_propagated_and_never_returns_tentative_success():
    class ConflictJournal(Journal):
        async def commit(self, batch, expected_version):
            self.calls.append("journal")
            raise JournalError(RunErrorCode.JOURNAL_CONFLICT)

    journal_calls = []
    journal = ConflictJournal(journal_calls)
    coordinator_, _, _, _, _ = coordinator(responses=[ModelResponse("answer")], journal=journal)
    with pytest.raises(JournalError) as error:
        run(coordinator_.advance(initial(), submit()))
    assert error.value.failure.code is RunErrorCode.JOURNAL_CONFLICT
    assert journal_calls == ["journal"]


@pytest.mark.parametrize(
    "field",
    [
        "type",
        "version",
        "sequence",
        "session",
        "state",
        "events",
        "event_sequence",
        "transcript",
        "transcript_sequence",
        "transcript_content",
        "summaries",
        "submission",
    ],
)
def test_malformed_journal_receipt_is_rejected_without_returning_split_cursor(field):
    class CorruptJournal(Journal):
        async def commit(self, batch, expected_version):
            receipt = await super().commit(batch, expected_version)
            if field == "type":
                return object()
            if field == "version":
                return replace(receipt, version=99)
            if field == "sequence":
                return replace(receipt, sequence=999)
            if field == "session":
                return replace(receipt, session_id=SessionId("other"))
            if field == "state":
                return replace(receipt, state=replace(batch.state, version=batch.state.version + 1))
            if field == "events":
                return replace(receipt, events=())
            if field == "event_sequence":
                changed = replace(receipt.events[0], sequence=999)
                return replace(receipt, events=(changed, *receipt.events[1:]))
            if field == "transcript":
                return replace(receipt, transcript=())
            if field == "transcript_sequence":
                changed = replace(receipt.transcript[0], sequence=999)
                return replace(receipt, transcript=(changed, *receipt.transcript[1:]))
            if field == "transcript_content":
                changed = replace(receipt.transcript[0], content="corrupt")
                return replace(receipt, transcript=(changed, *receipt.transcript[1:]))
            if field == "summaries":
                return replace(receipt, summaries=())
            return replace(receipt, submission=object())

    journal = CorruptJournal([])
    checkpoint = ConversationSummaryCheckpoint(SESSION, 1, "v1", PROFILE, "summary")
    coordinator_, _, _, _, _ = coordinator(
        responses=[ModelResponse("answer")], journal=journal, checkpoint=checkpoint
    )
    with pytest.raises(JournalError) as error:
        run(coordinator_.advance(initial(), submit()))
    assert error.value.failure.code is RunErrorCode.INVARIANT_VIOLATION


def test_journal_failure_leaves_readable_artifact_for_adapter_orphan_policy():
    class ConflictJournal(Journal):
        commits = 0

        async def commit(self, batch, expected_version):
            self.commits += 1
            if self.commits == 1:
                return await super().commit(batch, expected_version)
            raise JournalError(RunErrorCode.JOURNAL_CONFLICT)

    raw = ToolResult(
        ToolResultStatus.SUCCESS,
        inline_content=InlineContent({"payload": "x" * 80}, "application/json"),
    )
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    journal = ConflictJournal([])
    coordinator_, _, _, _, repository = coordinator(
        responses=[response],
        specs=(ToolSpec("read", "read", "{}"),),
        tool_results=[raw],
        journal=journal,
    )

    class RecordingExternalizer:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.refs = ()

        async def externalize(self, content, media_type, name=None):
            result = await self.wrapped.externalize(content, media_type, name)
            self.refs = result.artifacts
            return result

    recorder = RecordingExternalizer(coordinator_._artifacts)
    coordinator_._artifacts = recorder
    durable = run(coordinator_.advance(initial(), submit())).state
    with pytest.raises(JournalError):
        run(
            coordinator_.advance(
                durable, AdvanceRun(RUN, SESSION, context=command_context("tools"))
            )
        )
    assert len(recorder.refs) == 1
    assert run(repository.stat(recorder.refs[0].artifact_id)) == recorder.refs[0]


def test_failed_model_does_not_persist_a_new_summary_checkpoint():
    checkpoint = ConversationSummaryCheckpoint(SESSION, 1, "v1", PROFILE, "summary")
    coordinator_, _, journal, _, _ = coordinator(
        responses=[RuntimeError("provider detail")], checkpoint=checkpoint
    )
    outcome = run(coordinator_.advance(initial(), submit()))
    assert outcome.state.failure == RunFailure(RunErrorCode.MODEL_FAILED)
    assert journal.batches[0][0].summary is None


def test_checkpoint_is_attached_only_to_the_single_successful_commit():
    checkpoint = ConversationSummaryCheckpoint(SESSION, 1, "v1", PROFILE, "summary")
    coordinator_, _, journal, _, _ = coordinator(
        responses=[ModelResponse("answer")], checkpoint=checkpoint
    )
    run(coordinator_.advance(initial(), submit()))
    assert len(journal.batches) == 1
    assert journal.batches[0][0].summary == checkpoint


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (ToolResultStatus.FAILED, RunErrorCode.TOOL_FAILED),
        (ToolResultStatus.TIMED_OUT, RunErrorCode.TOOL_TIMED_OUT),
        (ToolResultStatus.DENIED, None),
    ],
)
def test_tool_failures_use_stable_classification(status, code):
    response = ModelResponse(tool_calls=(ModelToolCall("call", "read", "{}"),))
    result = ToolResult(status, failure=RunFailure(code) if code is not None else None)
    coordinator_, _, _, _, _ = coordinator(
        responses=[response], specs=(ToolSpec("read", "read", "{}"),), tool_results=[result]
    )
    durable = run(coordinator_.advance(initial(), submit())).state
    state = run(
        coordinator_.advance(durable, AdvanceRun(RUN, SESSION, context=command_context("tools")))
    ).state
    assert state.current_step is None
