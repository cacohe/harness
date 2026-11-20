import asyncio
from dataclasses import replace

from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.summarization import SummaryCheckpointService
from src.adapters.context.tokenizers import ExactTokenCounter
from src.core.artifacts import ArtifactRef
from src.core.context import (
    ContextBudget,
    ContextRequest,
    ConversationInteraction,
    ConversationPage,
)
from src.core.ids import ProfileId, SessionId
from src.core.journal import ConversationSummaryCheckpoint
from src.core.model import ModelMessage, ModelRole


def _request(**changes) -> ContextRequest:
    values = {
        "session_id": SessionId("session"),
        "profile_id": ProfileId("profile"),
        "policy_version": "policy-v1",
        "system_message": ModelMessage(ModelRole.SYSTEM, "system"),
        "current_input": ModelMessage(ModelRole.USER, "current"),
    }
    values.update(changes)
    return ContextRequest(**values)


class Compactor:
    def __init__(self, checkpoint=None, error=None):
        self.checkpoint = checkpoint
        self.error = error
        self.calls = []

    async def compact(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.checkpoint


class Reader:
    def __init__(self, records):
        self.records = records

    async def read_page(self, session_id, before_sequence, limit):
        records = tuple(
            record
            for record in self.records
            if before_sequence is None or record.sequence_through < before_sequence
        )
        page = records[:limit]
        return ConversationPage(page, page[-1].sequence_from if len(records) > len(page) else None)


def test_reuse_requires_every_checkpoint_key_component() -> None:
    request = _request()
    exact = ConversationSummaryCheckpoint(
        request.session_id, 12, request.policy_version, request.profile_id, "exact"
    )
    candidates = (
        replace(exact, covered_through_sequence=11, summary="sequence"),
        replace(exact, policy_version="policy-v2", summary="policy"),
        replace(exact, profile_id=ProfileId("other"), summary="profile"),
        exact,
    )
    service = SummaryCheckpointService(Compactor())
    assert service.find_reusable(candidates, request, 12) == exact
    assert service.find_reusable(candidates, request, 13) is None
    assert service.find_reusable(candidates, replace(request, policy_version="v3"), 12) is None


def test_compactor_failure_retains_original_records() -> None:
    records = (
        ConversationInteraction(1, 1, (ModelMessage(ModelRole.ASSISTANT, "one"),)),
        ConversationInteraction(2, 2, (ModelMessage(ModelRole.USER, "two"),)),
    )
    service = SummaryCheckpointService(Compactor(error=RuntimeError("provider secret")))
    outcome = asyncio.run(service.compact_or_retain(_request(), records))
    object.__setattr__(records[0], "sequence_from", 99)
    assert outcome.checkpoint is None
    assert outcome.retained[0].sequence_from == 1
    assert outcome.retained[1] == records[1]


def test_new_checkpoint_is_only_a_candidate_until_journal_commit() -> None:
    request = _request()
    records = (ConversationInteraction(4, 8, (ModelMessage(ModelRole.ASSISTANT, "history"),)),)
    candidate = ConversationSummaryCheckpoint(
        request.session_id,
        8,
        request.policy_version,
        request.profile_id,
        "summary",
        covered_from_sequence=4,
    )
    service = SummaryCheckpointService(Compactor(candidate))
    outcome = asyncio.run(service.compact_or_retain(request, records))
    assert outcome.checkpoint == candidate
    assert outcome.retained == records
    assert service.find_reusable((), request, 8) is None
    assert service.find_reusable((candidate,), request, 8) == candidate


def test_build_invokes_compactor_but_defers_candidate_until_a_trusted_request() -> None:
    request = _request()
    records = tuple(
        ConversationInteraction(index, index, (ModelMessage(ModelRole.ASSISTANT, "X" * 120),))
        for index in range(3, 0, -1)
    )
    digest = "b" * 64
    ref = ArtifactRef(f"sha256:{digest}", "application/json", 20, digest)
    candidate = ConversationSummaryCheckpoint(
        request.session_id,
        3,
        request.policy_version,
        request.profile_id,
        "compacted",
        artifacts=(ref,),
    )
    compactor = Compactor(candidate)
    provider = BoundedContextProvider(
        Reader(records),
        ExactTokenCounter(),
        page_size=2,
        summary_service=SummaryCheckpointService(compactor),
    )
    window = asyncio.run(provider.build(request, ContextBudget(127, 1, 0)))
    assert len(compactor.calls) == 1
    assert window.new_checkpoint == candidate
    rendered = "\n".join(message.content or "" for message in window.messages)
    assert "compacted" not in rendered
    assert ref.artifact_id not in rendered
    assert request.summary_checkpoints == ()


def test_build_compactor_failure_keeps_raw_records_authoritative() -> None:
    records = tuple(
        ConversationInteraction(index, index, (ModelMessage(ModelRole.ASSISTANT, "X" * 40),))
        for index in range(2, 0, -1)
    )
    provider = BoundedContextProvider(
        Reader(records),
        ExactTokenCounter(),
        page_size=1,
        summary_service=SummaryCheckpointService(Compactor(error=RuntimeError("secret"))),
    )
    window = asyncio.run(provider.build(_request(), ContextBudget(40, 1, 0)))
    assert window.new_checkpoint is None
    assert window.omitted[0].item_count == 2


def test_invalid_compactor_binding_is_rejected_and_raw_records_remain() -> None:
    request = _request()
    records = (ConversationInteraction(1, 3, (ModelMessage(ModelRole.ASSISTANT, "history"),)),)
    invalid = ConversationSummaryCheckpoint(
        request.session_id, 3, "other-policy", request.profile_id, "summary"
    )
    outcome = asyncio.run(
        SummaryCheckpointService(Compactor(invalid)).compact_or_retain(request, records)
    )
    assert outcome.checkpoint is None
    assert outcome.retained == records


def test_compaction_accumulates_the_latest_durable_summary() -> None:
    request = _request()
    prior = ConversationSummaryCheckpoint(
        request.session_id,
        5,
        request.policy_version,
        request.profile_id,
        "prior summary",
        covered_from_sequence=1,
    )
    records = (
        ConversationInteraction(6, 6, (ModelMessage(ModelRole.USER, "new"),)),
        ConversationInteraction(7, 7, (ModelMessage(ModelRole.ASSISTANT, "answer"),)),
    )
    candidate = ConversationSummaryCheckpoint(
        request.session_id,
        7,
        request.policy_version,
        request.profile_id,
        "cumulative summary",
        covered_from_sequence=1,
    )
    compactor = Compactor(candidate)

    outcome = asyncio.run(
        SummaryCheckpointService(compactor).compact_or_retain(
            request, records, prior_checkpoint=prior
        )
    )

    assert outcome.checkpoint == candidate
    assert len(compactor.calls) == 1
    assert compactor.calls[0].prior_checkpoint == prior


def test_compaction_rejects_a_gap_after_the_prior_checkpoint() -> None:
    prior = ConversationSummaryCheckpoint(
        SessionId("session"),
        27,
        "policy-v1",
        ProfileId("profile"),
        "summary-24-27",
        covered_from_sequence=24,
    )
    candidate = ConversationSummaryCheckpoint(
        SessionId("session"),
        38,
        "policy-v1",
        ProfileId("profile"),
        "invalid-gap",
        covered_from_sequence=24,
    )
    compactor = Compactor(candidate)
    service = SummaryCheckpointService(compactor)
    records = tuple(
        ConversationInteraction(
            index,
            index,
            (ModelMessage(ModelRole.ASSISTANT, f"history-{index}"),),
        )
        for index in range(35, 39)
    )

    outcome = asyncio.run(
        service.compact_or_retain(_request(), records, prior_checkpoint=prior)
    )

    assert outcome.checkpoint is None
    assert compactor.calls == []
