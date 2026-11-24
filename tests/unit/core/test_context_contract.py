from dataclasses import replace

import pytest
from src.core.artifacts.contracts import ArtifactRef
from src.core.context.contracts import (
    ContextBudget,
    ContextRequest,
    ContextWindow,
    ConversationInteraction,
    ConversationPage,
    EstimateQuality,
    OmittedContextRef,
    TokenEstimate,
    snapshot_summary_checkpoint,
)
from src.core.context.model import ModelMessage, ModelRole
from src.core.ids import ProfileId, SessionId
from src.core.persistence.journal import ConversationSummaryCheckpoint


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


def test_budget_calculates_input_allowance() -> None:
    budget = ContextBudget(8_192, 1_024, 256)
    assert budget.input_allowance_tokens == 6_912


@pytest.mark.parametrize(
    "values",
    [
        (0, 0, 0),
        (-1, 0, 0),
        (100, -1, 0),
        (100, 0, -1),
        (100, 100, 0),
        (100, 40, 60),
        (True, 0, 0),
        (100, False, 0),
        (100, 0, True),
    ],
)
def test_budget_rejects_invalid_or_exhausted_input_space(values) -> None:
    with pytest.raises(ValueError):
        ContextBudget(*values)


@pytest.mark.parametrize("quality", list(EstimateQuality))
def test_estimate_explicitly_identifies_quality(quality: EstimateQuality) -> None:
    estimate = TokenEstimate(12, quality)
    assert estimate.quality is quality


@pytest.mark.parametrize("count", [-1, True, 1.0])
def test_estimate_requires_an_exact_nonnegative_integer(count) -> None:
    with pytest.raises(ValueError):
        TokenEstimate(count, EstimateQuality.EXACT)


def test_window_rejects_an_estimate_above_the_input_allowance() -> None:
    budget = ContextBudget(10, 2, 1)
    with pytest.raises(ValueError):
        ContextWindow(
            messages=(ModelMessage(ModelRole.USER, "current"),),
            tools=(),
            estimate=TokenEstimate(8, EstimateQuality.CONSERVATIVE),
            budget=budget,
            omitted=(),
            policy_version="policy-v1",
        )


def test_window_snapshots_omissions_and_candidate_checkpoint() -> None:
    request = _request()
    checkpoint = ConversationSummaryCheckpoint(
        request.session_id, 9, request.policy_version, request.profile_id, "summary"
    )
    omission = OmittedContextRef("transcript", 1, 9, 4)
    window = ContextWindow(
        messages=(request.system_message, request.current_input),
        tools=(),
        estimate=TokenEstimate(7, EstimateQuality.EXACT),
        budget=ContextBudget(10, 2, 1),
        omitted=(omission,),
        policy_version=request.policy_version,
        new_checkpoint=checkpoint,
    )
    object.__setattr__(omission, "item_count", 99)
    object.__setattr__(checkpoint, "summary", "changed")
    assert window.omitted[0].item_count == 4
    assert window.new_checkpoint is not None
    assert window.new_checkpoint.summary == "summary"


def test_request_rejects_checkpoint_from_another_policy_or_profile() -> None:
    request = _request()
    checkpoint = ConversationSummaryCheckpoint(
        request.session_id, 4, request.policy_version, request.profile_id, "summary"
    )
    snapshots = replace(request, summary_checkpoints=(checkpoint,)).summary_checkpoints
    assert snapshots == (checkpoint,)
    assert replace(checkpoint, session_id=SessionId("other")) not in snapshots
    assert replace(checkpoint, policy_version="policy-v2") not in snapshots
    assert replace(checkpoint, profile_id=ProfileId("other")) not in snapshots


def test_request_rejects_non_tool_messages_in_the_open_pair_slot() -> None:
    with pytest.raises(ValueError):
        _request(open_tool_messages=(ModelMessage(ModelRole.ASSISTANT, "plain"),))


def test_conversation_page_rejects_remaining_aggregate_that_disagrees_with_cursor() -> None:
    interaction = ConversationInteraction(
        1_000,
        1_001,
        (ModelMessage(ModelRole.ASSISTANT, "newest"),),
    )
    with pytest.raises(ValueError):
        ConversationPage(
            (interaction,),
            100,
            remaining_omission=OmittedContextRef("transcript", 1, 999, 999),
        )
    with pytest.raises(ValueError):
        ConversationPage(
            (interaction,),
            1_000,
            remaining_omission=OmittedContextRef("transcript", 1, 999, 1_000),
        )


def test_checkpoint_snapshot_uses_only_trusted_fixed_fields() -> None:
    class Trap:
        reads = 0

        @property
        def value(self):
            Trap.reads += 1
            raise RuntimeError("must not execute")

    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"), 4, "policy-v1", ProfileId("profile"), "summary"
    )
    object.__setattr__(checkpoint, "session_id", Trap())
    object.__setattr__(checkpoint, "__dataclass_fields__", Trap())
    with pytest.raises(ValueError):
        snapshot_summary_checkpoint(checkpoint)
    assert Trap.reads == 0

    missing = ConversationSummaryCheckpoint(
        SessionId("session"), 4, "policy-v1", ProfileId("profile"), "summary"
    )
    object.__delattr__(missing, "profile_id")
    with pytest.raises(ValueError):
        snapshot_summary_checkpoint(missing)


def test_checkpoint_snapshot_copies_structured_artifact_refs() -> None:
    digest = "a" * 64
    ref = ArtifactRef(f"sha256:{digest}", "application/json", 2, digest)
    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"),
        4,
        "policy-v1",
        ProfileId("profile"),
        "two keys",
        artifacts=(ref,),
    )
    snapshot = snapshot_summary_checkpoint(checkpoint)
    assert snapshot is not None
    object.__setattr__(ref, "size_bytes", 99)
    assert snapshot.artifacts[0].size_bytes == 2
