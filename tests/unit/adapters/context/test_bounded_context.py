import asyncio
from bisect import bisect_left, bisect_right

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
    OmittedContextRef,
)
from src.core.ids import ProfileId, SessionId
from src.core.journal import ConversationSummaryCheckpoint
from src.core.model import ModelMessage, ModelRole, ModelToolCall
from src.core.tools import ToolSpec


class Reader:
    def __init__(self, items: tuple[ConversationInteraction, ...]):
        self.items = tuple(sorted(items, key=lambda item: item.sequence_through))
        self.calls: list[tuple[int | None, int]] = []
        self.aggregate_calls: list[tuple[int, int | None, int | None]] = []

    async def read_page(self, session_id, before_sequence, limit):
        assert session_id == SessionId("session")
        self.calls.append((before_sequence, limit))
        end = (
            len(self.items)
            if before_sequence is None
            else bisect_left(self.items, before_sequence, key=lambda item: item.sequence_through)
        )
        start = max(0, end - limit)
        page = tuple(reversed(self.items[start:end]))
        more = start > 0
        return ConversationPage(page, page[-1].sequence_from if more else None)

    async def summarize_before(
        self,
        session_id,
        before_sequence,
        excluded_from_sequence=None,
        excluded_through_sequence=None,
    ):
        assert session_id == SessionId("session")
        self.aggregate_calls.append(
            (before_sequence, excluded_from_sequence, excluded_through_sequence)
        )
        end = bisect_left(self.items, before_sequence, key=lambda item: item.sequence_through)

        def omission(start: int, stop: int):
            if start >= stop:
                return None
            return OmittedContextRef(
                "transcript",
                self.items[start].sequence_from,
                self.items[stop - 1].sequence_through,
                stop - start,
            )

        if excluded_from_sequence is None:
            aggregate = omission(0, end)
            return (aggregate,) if aggregate is not None else ()
        assert excluded_through_sequence is not None
        high_start = bisect_right(
            self.items,
            excluded_through_sequence,
            hi=end,
            key=lambda item: item.sequence_through,
        )
        low_end = bisect_left(
            self.items,
            excluded_from_sequence,
            hi=end,
            key=lambda item: item.sequence_through,
        )
        return tuple(
            item for item in (omission(high_start, end), omission(0, low_end)) if item is not None
        )


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
        summary_checkpoints=(
            ConversationSummaryCheckpoint(
                SessionId("session"), 1, "policy-v1", ProfileId("profile"), "SUM"
            ),
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
    assert (window.omitted[0].first_sequence, window.omitted[0].last_sequence) == (2, 3)
    assert window.omitted[0].item_count == 2
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


def test_ten_thousand_records_with_partial_checkpoint_remain_bounded_and_deterministic() -> None:
    records = tuple(_interaction(index, f"history-{index:05d}") for index in range(1, 10_001))
    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"),
        6_000,
        "policy-v1",
        ProfileId("profile"),
        "middle-summary",
        covered_from_sequence=4_001,
    )
    reader = Reader(records)
    provider = BoundedContextProvider(reader, ExactTokenCounter(), page_size=127)
    request = _request(summary_checkpoints=(checkpoint,))

    first = asyncio.run(provider.build(request, ContextBudget(128, 8, 8)))
    second = asyncio.run(provider.build(request, ContextBudget(128, 8, 8)))

    assert first == second
    assert first.estimate.tokens <= 112
    transcript = [item for item in first.omitted if item.source == "transcript"]
    assert [(item.first_sequence, item.last_sequence) for item in transcript] == [
        (1, 4_000),
        (6_001, 9_996),
    ]
    assert len(reader.calls) <= 4
    assert len(reader.aggregate_calls) == 2


def test_tools_are_counted_in_the_budget() -> None:
    tool = ToolSpec("read", "DDD", "{}")
    request = _request(tools=(tool,))
    provider = BoundedContextProvider(Reader(()), ExactTokenCounter(), page_size=2)
    with pytest.raises(ContextBudgetExceeded):
        asyncio.run(provider.build(request, ContextBudget(24, 1, 0)))
    assert asyncio.run(provider.build(request, ContextBudget(25, 1, 0))).tools == (tool,)


def test_durable_summary_excludes_covered_raw_history() -> None:
    records = tuple(_interaction(index, f"raw-{index}") for index in range(1, 5))
    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"), 3, "policy-v1", ProfileId("profile"), "summary-1-3"
    )
    window = asyncio.run(
        BoundedContextProvider(Reader(records), ExactTokenCounter(), page_size=2).build(
            _request(summary_checkpoints=(checkpoint,)), ContextBudget(128, 1, 0)
        )
    )
    contents = [message.content for message in window.messages]
    assert "summary-1-3" in contents
    assert "raw-4" in contents
    assert not {"raw-1", "raw-2", "raw-3"} & set(contents)


@pytest.mark.parametrize("page_size", [2, 10])
def test_partial_range_summary_keeps_older_raw_in_time_order(page_size: int) -> None:
    records = tuple(_interaction(index, f"raw-{index}") for index in range(1, 8))
    reader = Reader(records)
    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"),
        5,
        "policy-v1",
        ProfileId("profile"),
        "summary-3-5",
        covered_from_sequence=3,
    )
    window = asyncio.run(
        BoundedContextProvider(reader, ExactTokenCounter(), page_size=page_size).build(
            _request(summary_checkpoints=(checkpoint,)), ContextBudget(128, 1, 0)
        )
    )
    assert [message.content for message in window.messages] == [
        "SS",
        "raw-1",
        "raw-2",
        "summary-3-5",
        "raw-6",
        "raw-7",
        "UUU",
    ]
    if page_size == 2:
        assert len(reader.calls) == 4


def test_older_raw_after_partial_summary_is_precisely_omitted_when_too_large() -> None:
    records = (
        _interaction(7, "n"),
        _interaction(6, "n"),
        _interaction(5, "covered"),
        _interaction(4, "covered"),
        _interaction(3, "covered"),
        _interaction(2, "X" * 20),
        _interaction(1, "X" * 20),
    )
    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"),
        5,
        "policy-v1",
        ProfileId("profile"),
        "sum",
        covered_from_sequence=3,
    )
    window = asyncio.run(
        BoundedContextProvider(Reader(records), ExactTokenCounter(), page_size=2).build(
            _request(summary_checkpoints=(checkpoint,)), ContextBudget(50, 1, 0)
        )
    )
    contents = [message.content for message in window.messages]
    assert contents == ["SS", "sum", "n", "n", "UUU"]
    transcript_omissions = [item for item in window.omitted if item.source == "transcript"]
    assert len(transcript_omissions) == 1
    assert (
        transcript_omissions[0].first_sequence,
        transcript_omissions[0].last_sequence,
        transcript_omissions[0].item_count,
    ) == (1, 2, 2)


def test_persisted_candidate_does_not_hide_history_before_its_source_range() -> None:
    candidate = ConversationSummaryCheckpoint(
        SessionId("session"),
        5,
        "policy-v1",
        ProfileId("profile"),
        "persisted-candidate",
        covered_from_sequence=3,
    )
    records = tuple(_interaction(index, f"raw-{index}") for index in range(1, 7))
    window = asyncio.run(
        BoundedContextProvider(Reader(records), ExactTokenCounter(), page_size=2).build(
            _request(summary_checkpoints=(candidate,)), ContextBudget(128, 1, 0)
        )
    )
    contents = [message.content for message in window.messages]
    assert contents.index("raw-1") < contents.index("persisted-candidate")
    assert contents.index("raw-2") < contents.index("persisted-candidate")
    assert contents.index("persisted-candidate") < contents.index("raw-6")


def test_checkpoint_boundary_cannot_split_an_interaction() -> None:
    checkpoint = ConversationSummaryCheckpoint(
        SessionId("session"),
        5,
        "policy-v1",
        ProfileId("profile"),
        "summary",
        covered_from_sequence=3,
    )
    interaction = ConversationInteraction(2, 4, (ModelMessage(ModelRole.ASSISTANT, "overlap"),))
    with pytest.raises(ValueError, match="splits"):
        asyncio.run(
            BoundedContextProvider(Reader((interaction,)), ExactTokenCounter(), page_size=1).build(
                _request(summary_checkpoints=(checkpoint,)), ContextBudget(64, 1, 0)
            )
        )


@pytest.mark.parametrize(
    "checkpoint",
    [
        ConversationSummaryCheckpoint(
            SessionId("other"), 1, "policy-v1", ProfileId("profile"), "wrong-session"
        ),
        ConversationSummaryCheckpoint(
            SessionId("session"), 1, "policy-v2", ProfileId("profile"), "wrong-policy"
        ),
        ConversationSummaryCheckpoint(
            SessionId("session"), 1, "policy-v1", ProfileId("other"), "wrong-profile"
        ),
    ],
)
def test_mismatched_durable_summary_is_not_reused(checkpoint) -> None:
    window = asyncio.run(
        BoundedContextProvider(
            Reader((_interaction(1, "raw-1"),)), ExactTokenCounter(), page_size=1
        ).build(_request(summary_checkpoints=(checkpoint,)), ContextBudget(64, 1, 0))
    )
    contents = [message.content for message in window.messages]
    assert "raw-1" in contents
    assert checkpoint.summary not in contents


def test_history_selection_is_a_contiguous_newest_suffix() -> None:
    records = (_interaction(2, "X" * 80), _interaction(1, "old"))
    window = asyncio.run(
        BoundedContextProvider(Reader(records), ExactTokenCounter(), page_size=1).build(
            _request(), ContextBudget(40, 1, 0)
        )
    )
    contents = [message.content for message in window.messages]
    assert "X" * 80 not in contents
    assert "old" not in contents
    assert window.omitted[0].item_count == 2


def test_conservative_fallback_applies_safety_factor() -> None:
    counter = ConservativeTokenCounter(chars_per_token=4, safety_factor=1.5)
    estimate = counter.count((ModelMessage(ModelRole.USER, "12345678"),), ())
    assert estimate.tokens >= 3
    assert estimate.quality is EstimateQuality.CONSERVATIVE
