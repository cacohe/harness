import asyncio
from dataclasses import replace

from src.adapters.context.summarization import SummaryCheckpointService
from src.core.context import ContextRequest, ConversationInteraction
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
