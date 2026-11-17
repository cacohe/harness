import asyncio

import pytest
from src.adapters.context.bounded import BoundedContextProvider
from src.adapters.context.tokenizers import ConservativeTokenCounter, ExactTokenCounter
from src.core.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextRequest,
    ConversationInteraction,
    ConversationPage,
    EstimateQuality,
)
from src.core.ids import ProfileId, SessionId
from src.core.journal import ConversationSummaryCheckpoint
from src.core.model import ModelMessage, ModelRole, ModelToolCall
from src.core.tools import ToolSpec


class Reader:
    def __init__(self, items: tuple[ConversationInteraction, ...]):
        self.items = tuple(sorted(items, key=lambda item: item.sequence_through, reverse=True))
        self.calls: list[tuple[int | None, int]] = []

    async def read_page(self, session_id, before_sequence, limit):
        assert session_id == SessionId("session")
        self.calls.append((before_sequence, limit))
        eligible = [
            item
            for item in self.items
            if before_sequence is None or item.sequence_through < before_sequence
        ]
        page = tuple(eligible[:limit])
        more = len(eligible) > len(page)
        return ConversationPage(page, page[-1].sequence_from if more else None)


def _request(**changes) -> ContextRequest:
    values = {
        "session_id": SessionId("session"),
        "profile_id": ProfileId("profile"),
        "policy_version": "policy-v1",
        "system_message": ModelMessage(ModelRole.SYSTEM, "SS"),
        "current_input": ModelMessage(ModelRole.USER, "UUU"),
    }
    values.update(changes)
    return ContextRequest(**values)


def _interaction(sequence: int, text: str) -> ConversationInteraction:
    return ConversationInteraction(
        sequence,
        sequence,
        (ModelMessage(ModelRole.ASSISTANT, text),),
    )


def test_required_context_exactly_fits_and_one_token_less_fails() -> None:
    provider = BoundedContextProvider(Reader(()), ExactTokenCounter(), page_size=4)
    request = _request()
    exact = asyncio.run(provider.build(request, ContextBudget(16, 1, 0)))
    assert exact.estimate.tokens == 15
    assert exact.messages == (request.system_message, request.current_input)

    with pytest.raises(ContextBudgetExceeded):
        asyncio.run(provider.build(request, ContextBudget(15, 1, 0)))


def test_open_tool_pair_is_mandatory_and_never_split() -> None:
    call = ModelToolCall("call", "read", "{}")
    open_pair = (
        ModelMessage(ModelRole.ASSISTANT, tool_calls=(call,)),
        ModelMessage(ModelRole.TOOL, "ok", tool_call_id="call"),
    )
    request = _request(open_tool_messages=open_pair)
    provider = BoundedContextProvider(Reader(()), ExactTokenCounter(), page_size=2)

    window = asyncio.run(provider.build(request, ContextBudget(45, 1, 0)))
    assert window.messages == (
        request.system_message,
        *open_pair,
        request.current_input,
    )
    with pytest.raises(ContextBudgetExceeded):
        asyncio.run(provider.build(request, ContextBudget(44, 1, 0)))


def test_selection_is_newest_first_but_output_is_chronological() -> None:
    reader = Reader(tuple(_interaction(index, str(index) * 3) for index in range(1, 6)))
    request = _request(
        summary_checkpoint=ConversationSummaryCheckpoint(
            SessionId("session"), 1, "policy-v1", ProfileId("profile"), "SUM"
        ),
        memory_messages=(ModelMessage(ModelRole.SYSTEM, "MEM"),),
        skill_messages=(ModelMessage(ModelRole.SYSTEM, "SKL"),),
    )
    provider = BoundedContextProvider(reader, ExactTokenCounter(), page_size=2)

    window = asyncio.run(provider.build(request, ContextBudget(40, 1, 0)))

    contents = [message.content for message in window.messages]
    assert contents == ["SS", "444", "555", "UUU"]
    assert window.estimate.tokens <= window.budget.input_allowance_tokens
    assert len(window.omitted) == 4
    assert (window.omitted[0].first_sequence, window.omitted[0].last_sequence) == (1, 3)
    assert window.omitted[0].item_count == 3
    assert [item.source for item in window.omitted] == [
        "transcript",
        "summary",
        "memory",
        "skills",
    ]


def test_complete_tool_interaction_is_selected_atomically() -> None:
    call = ModelToolCall("c1", "read", "{}")
    tool_interaction = ConversationInteraction(
        4,
        5,
        (
            ModelMessage(ModelRole.ASSISTANT, tool_calls=(call,)),
            ModelMessage(ModelRole.TOOL, "result", tool_call_id="c1"),
        ),
    )
    provider = BoundedContextProvider(Reader((tool_interaction,)), ExactTokenCounter(), page_size=2)
    request = _request()

    excluded = asyncio.run(provider.build(request, ContextBudget(44, 1, 0)))
    included = asyncio.run(provider.build(request, ContextBudget(45, 1, 0)))
    assert all(message.tool_call_id != "c1" for message in excluded.messages)
    assert [message.role for message in included.messages[1:3]] == [
        ModelRole.ASSISTANT,
        ModelRole.TOOL,
    ]


def test_artifact_history_contains_only_bounded_summary_and_reference() -> None:
    content = "artifact sha256:" + "a" * 64 + " — 2 keys"
    provider = BoundedContextProvider(
        Reader(
            (
                ConversationInteraction(
                    1,
                    1,
                    (ModelMessage(ModelRole.ASSISTANT, content),),
                ),
            )
        ),
        ExactTokenCounter(),
        page_size=2,
    )
    window = asyncio.run(provider.build(_request(), ContextBudget(200, 1, 0)))
    assert content in [message.content for message in window.messages]
    assert "payload" not in "".join(message.content or "" for message in window.messages)


def test_ten_thousand_records_are_paged_and_output_remains_bounded() -> None:
    records = tuple(_interaction(index, f"history-{index:05d}") for index in range(1, 10_001))
    reader = Reader(records)
    provider = BoundedContextProvider(reader, ExactTokenCounter(), page_size=127)

    first = asyncio.run(provider.build(_request(), ContextBudget(128, 8, 8)))
    second = asyncio.run(provider.build(_request(), ContextBudget(128, 8, 8)))

    assert first == second
    assert first.estimate.tokens <= 112
    assert len(first.messages) < 20
    assert len(reader.calls) > 2
    assert all(limit == 127 for _, limit in reader.calls)
    assert first.omitted[0].item_count > 9_990


def test_tools_are_counted_in_the_budget() -> None:
    tool = ToolSpec("read", "DDD", "{}")
    request = _request(tools=(tool,))
    provider = BoundedContextProvider(Reader(()), ExactTokenCounter(), page_size=2)
    with pytest.raises(ContextBudgetExceeded):
        asyncio.run(provider.build(request, ContextBudget(24, 1, 0)))
    assert asyncio.run(provider.build(request, ContextBudget(25, 1, 0))).tools == (tool,)


def test_uncommitted_checkpoint_is_returned_only_as_a_candidate() -> None:
    candidate = ConversationSummaryCheckpoint(
        SessionId("session"), 7, "policy-v1", ProfileId("profile"), "candidate"
    )
    window = asyncio.run(
        BoundedContextProvider(Reader(()), ExactTokenCounter(), page_size=2).build(
            _request(new_checkpoint=candidate), ContextBudget(32, 1, 0)
        )
    )
    assert window.new_checkpoint == candidate
    assert "candidate" not in [message.content for message in window.messages]


def test_conservative_fallback_applies_safety_factor() -> None:
    counter = ConservativeTokenCounter(chars_per_token=4, safety_factor=1.5)
    estimate = counter.count((ModelMessage(ModelRole.USER, "12345678"),), ())
    assert estimate.tokens >= 3
    assert estimate.quality is EstimateQuality.CONSERVATIVE
